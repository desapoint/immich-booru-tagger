"""
Main processor for the Immich Auto-Tagger service.
"""

import gc
import time
from typing import Dict, List, Optional
from .immich_client import ImmichClient
from .tagging_engine import create_tagging_engine
from .models import Asset, Tag, AssetProcessingResult, BatchProcessingResult
from .config import settings
from .logging import get_logger, MetricsLogger
from .performance_monitor import performance_monitor
from .failure_tracker import FailureTracker


CONTENT_RATINGS = ("general", "sensitive", "questionable", "explicit")


class ProcessorError(Exception):
    """Custom exception for processor errors."""
    pass


class ImmichAutoTagger:
    """Main processor for auto-tagging Immich assets."""
    
    def __init__(self):
        self.logger = get_logger("processor")
        self.metrics = MetricsLogger()
        self.immich_client = ImmichClient()
        self.tagging_engine = None
        self.processed_tag: Optional[Tag] = None
        self.processed_tags: Dict[str, Tag] = {}
        self.content_rating_tags: Dict[str, Dict[str, Tag]] = {}
        
        # Progress tracking (global and per-library)
        self.total_processed_assets = 0
        self.total_assigned_tags = 0
        self.library_metrics: Dict[str, Dict] = {}
        
        # Initialize failure tracking (will be set per library)
        self.failure_tracker = None
        self.library_failure_trackers: Dict[str, FailureTracker] = {}
        
        # Initialize tracking and managed tags for the first library.
        self.set_current_library(self.immich_client.current_library_name)

    @property
    def is_tagging_engine_loaded(self) -> bool:
        """Return whether the ONNX session is currently resident in memory."""
        return self.tagging_engine is not None

    def _get_tagging_engine(self):
        """Load the model on first use and reuse it until explicitly unloaded."""
        if self.tagging_engine is None:
            self.logger.info("📦 Loading ONNX model into memory")
            try:
                self.tagging_engine = create_tagging_engine()
            except Exception as e:
                self.logger.error(f"❌ Failed to load ONNX model: {e}")
                raise
        return self.tagging_engine

    def unload_tagging_engine(self) -> bool:
        """Release the active ONNX session while preserving its disk cache."""
        if self.tagging_engine is None:
            return False

        self.logger.info("📤 Unloading ONNX model from memory")
        tagging_engine = self.tagging_engine
        self.tagging_engine = None
        del tagging_engine
        gc.collect()
        self.logger.info("✅ ONNX model unloaded from memory")
        return True
    
    def _initialize_processed_tag(self):
        """Select or initialize the processed tag for the current library."""
        library_key = self.immich_client.api_key

        if library_key in self.processed_tags:
            self.processed_tag = self.processed_tags[library_key]
            return

        try:
            self.processed_tag = self.immich_client.get_or_create_tag(settings.processed_tag_name)
            self.processed_tags[library_key] = self.processed_tag
            self.logger.info(f"🏷️  Using processed tag: '{self.processed_tag.name}'")
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize processed tag: {str(e)}")
            raise ProcessorError(f"Failed to initialize processed tag: {e}")

    def _initialize_content_rating_tags(self):
        """Ensure rating hierarchy exists and migrate matching legacy tags."""
        library_key = self.immich_client.api_key

        if library_key in self.content_rating_tags:
            return

        try:
            # Capture only pre-existing flat tags before creating the hierarchy.
            existing_tags = self.immich_client.get_all_tags(use_cache=True)
            legacy_tags = {
                tag.path.casefold(): tag
                for tag in existing_tags
                if tag.parentId is None and tag.path.casefold() in CONTENT_RATINGS
            }

            parent = self.immich_client.get_or_create_tag(
                settings.content_rating_tag_name
            )
            rating_tags = {
                rating: self.immich_client.get_or_create_child_tag(parent, rating)
                for rating in CONTENT_RATINGS
            }
            self.content_rating_tags[library_key] = rating_tags

            self.logger.info(
                "🏷️  Using content-rating hierarchy: "
                f"'{settings.content_rating_tag_name}/"
                f"{{{', '.join(CONTENT_RATINGS)}}}'"
            )

            # Immich cannot rename or reparent a tag. Move associations to the
            # child tag and remove only a matching legacy flat rating tag.
            for rating, source_tag in legacy_tags.items():
                destination_tag = rating_tags[rating]
                try:
                    migrated_assets = self.immich_client.migrate_tag(
                        source=source_tag,
                        destination=destination_tag,
                    )
                    self.logger.info(
                        f"♻️  Migrated flat rating tag '{source_tag.path}' to "
                        f"'{destination_tag.path}' on {migrated_assets} assets"
                    )
                except Exception as e:
                    # The hierarchy is still usable for new predictions. A
                    # failed migration is safe to retry on the next launch.
                    self.logger.error(
                        f"❌ Failed to migrate flat rating tag "
                        f"'{source_tag.path}': {e}"
                    )
        except Exception as e:
            self.logger.error(
                f"❌ Failed to initialize content-rating hierarchy: {e}"
            )
            raise ProcessorError(
                f"Failed to initialize content-rating hierarchy: {e}"
            ) from e

    def _get_content_rating_tag(self, prediction_name: str) -> Optional[Tag]:
        """Resolve a raw model rating name to the current library's child tag."""
        rating = prediction_name.strip().casefold()
        if rating not in CONTENT_RATINGS:
            return None
        return self.content_rating_tags.get(
            self.immich_client.api_key,
            {},
        ).get(rating)

    def _asset_is_processed(self, asset: Asset) -> bool:
        """Return whether an asset already carries this library's marker."""
        return bool(
            self.processed_tag
            and asset.tags
            and any(tag.id == self.processed_tag.id for tag in asset.tags)
        )

    def _failed_result(
        self,
        asset_id: str,
        start_time: float,
        error: Exception,
    ) -> AssetProcessingResult:
        """Create and record an asset-level processing failure."""
        self.metrics.metrics["failures"] += 1
        return AssetProcessingResult(
            asset_id=asset_id,
            success=False,
            error=str(error),
            processing_time=time.time() - start_time,
        )

    def _apply_predictions(
        self,
        asset: Asset,
        predictions,
        start_time: float,
    ) -> AssetProcessingResult:
        """Resolve model predictions, assign tags, and mark an asset complete."""
        result = AssetProcessingResult(asset_id=asset.id, success=False)

        try:
            tag_names = [
                prediction.name
                for prediction in predictions
                if self._get_content_rating_tag(prediction.name) is None
            ]
            tag_mapping = self.immich_client.get_or_create_tags_bulk(tag_names)

            tag_ids = []
            assigned_tag_ids = set()
            for prediction in predictions:
                tag_name = prediction.name
                tag = self._get_content_rating_tag(tag_name)
                if tag is None:
                    tag = tag_mapping.get(tag_name)
                if tag is None or tag.id in assigned_tag_ids:
                    continue

                assigned_tag_ids.add(tag.id)
                tag_ids.append(tag.id)
                result.tags_assigned.append(tag.path)

            if tag_ids:
                self.immich_client.tag_single_asset(asset.id, tag_ids)

            if self.processed_tag:
                self.immich_client.tag_single_asset(
                    asset.id,
                    [self.processed_tag.id],
                )

            result.success = True
            result.processing_time = time.time() - start_time
            self.metrics.metrics["assets_processed"] += 1
            self.metrics.metrics["tags_assigned"] += len(result.tags_assigned)
            self.metrics.metrics["processing_time"] += result.processing_time
            performance_monitor.record_asset_processed(result.processing_time)
            return result
        except Exception as e:
            return self._failed_result(asset.id, start_time, e)
    
    def process_asset(self, asset: Asset) -> AssetProcessingResult:
        """Process a single asset for tagging."""
        start_time = time.time()

        if asset.type != "IMAGE":
            return AssetProcessingResult(
                asset_id=asset.id,
                success=False,
                error=f"Unsupported asset type: {asset.type}",
            )

        if self._asset_is_processed(asset):
            self.logger.debug(f"⏭️  Skipping already processed asset: {asset.id}")
            return AssetProcessingResult(
                asset_id=asset.id,
                success=True,
                processing_time=time.time() - start_time,
            )

        try:
            image_data = self.immich_client.download_asset(asset.id, use_thumbnail=True)
            predictions = self._get_tagging_engine().predict_tags(image_data)
            return self._apply_predictions(asset, predictions, start_time)
        except Exception as e:
            return self._failed_result(asset.id, start_time, e)
    
    def process_batch(self, assets: List[Asset]) -> BatchProcessingResult:
        """Process a batch of assets with optimized bulk operations."""
        start_time = time.time()
        results = [None] * len(assets)
        
        # Pre-warm the tag cache before processing
        try:
            self.immich_client.get_all_tags(use_cache=True)
        except Exception as e:
            self.logger.warning(f"⚠️  Failed to pre-warm tag cache: {str(e)}")

        download_start_time = time.time()
        inference_items = []
        for index, asset in enumerate(assets):
            asset_start_time = time.time()

            if asset.type != "IMAGE":
                results[index] = AssetProcessingResult(
                    asset_id=asset.id,
                    success=False,
                    error=f"Unsupported asset type: {asset.type}",
                )
                continue

            if self._asset_is_processed(asset):
                results[index] = AssetProcessingResult(
                    asset_id=asset.id,
                    success=True,
                    processing_time=time.time() - asset_start_time,
                )
                continue

            try:
                image_data = self.immich_client.download_asset(
                    asset.id,
                    use_thumbnail=True,
                )
                inference_items.append(
                    (index, asset, image_data, asset_start_time)
                )
            except Exception as e:
                results[index] = self._failed_result(
                    asset.id,
                    asset_start_time,
                    e,
                )

        download_time = time.time() - download_start_time
        self.logger.info(
            "📥 Batch download complete: "
            f"{len(inference_items)}/{len(assets)} images ready "
            f"in {download_time:.1f}s"
        )

        if inference_items:
            try:
                tagging_engine = self._get_tagging_engine()
            except Exception as e:
                for index, asset, _, asset_start_time in inference_items:
                    results[index] = self._failed_result(
                        asset.id,
                        asset_start_time,
                        e,
                    )
            else:
                try:
                    prediction_batches = tagging_engine.predict_tags_batch(
                        [item[2] for item in inference_items]
                    )
                    if len(prediction_batches) != len(inference_items):
                        raise ProcessorError(
                            "Tagging engine returned an unexpected batch size"
                        )

                    for item, predictions in zip(
                        inference_items,
                        prediction_batches,
                    ):
                        index, asset, _, asset_start_time = item
                        results[index] = self._apply_predictions(
                            asset,
                            predictions,
                            asset_start_time,
                        )
                except Exception as e:
                    self.logger.warning(
                        "⚠️  Batched inference failed; retrying images "
                        f"individually: {e}"
                    )
                    for index, asset, image_data, asset_start_time in inference_items:
                        try:
                            predictions = tagging_engine.predict_tags(image_data)
                            results[index] = self._apply_predictions(
                                asset,
                                predictions,
                                asset_start_time,
                            )
                        except Exception as item_error:
                            results[index] = self._failed_result(
                                asset.id,
                                asset_start_time,
                                item_error,
                            )

        if any(result is None for result in results):
            raise ProcessorError("Batch processing did not produce every result")
        
        batch_time = time.time() - start_time
        
        # Process failure tracking for failed assets
        for i, result in enumerate(results):
            if not result.success and result.error:
                asset = assets[i]
                should_retry = self.failure_tracker.record_failure(asset.id)
                if not should_retry:
                    self.logger.warning(f"❌ Asset {asset.originalFileName} ({asset.id}) marked as permanently failed")
        
        # Calculate batch statistics
        successful = sum(1 for r in results if r.success)
        failed = sum(1 for r in results if not r.success and r.error)
        skipped = sum(1 for r in results if r.success and not r.tags_assigned)  # Already processed
        processed = sum(1 for r in results if r.success and r.tags_assigned)  # Newly processed
        total_tags_assigned = sum(len(r.tags_assigned) for r in results if r.success)
        
        batch_result = BatchProcessingResult(
            batch_size=len(assets),
            successful=successful,
            failed=failed,
            total_tags_assigned=total_tags_assigned,
            processing_time=batch_time,
            results=results
        )
        
        # Update totals (only count newly processed assets, not skipped ones)
        self.total_processed_assets += processed
        self.total_assigned_tags += total_tags_assigned
        
        # Update library-specific metrics
        current_library = self.immich_client.current_library_name
        if current_library in self.library_metrics:
            self.library_metrics[current_library]["processed_assets"] += processed
            self.library_metrics[current_library]["assigned_tags"] += total_tags_assigned
            self.library_metrics[current_library]["failed_assets"] += failed
        
        # Record performance metrics
        performance_monitor.record_batch_processed(batch_time)
        
        
        # Clean, focused logging with progress
        rate_per_second = len(assets) / batch_time if batch_time > 0 else 0
        
        # Create status message based on what happened
        if skipped > 0:
            status_msg = f"📊 Batch: {processed} processed, {skipped} already done"
            if failed > 0:
                status_msg += f", {failed} failed"
        else:
            status_msg = f"📊 Batch: {processed} processed"
            if failed > 0:
                status_msg += f", {failed} failed"
        
        self.logger.info(
            f"{status_msg} | "
            f"{total_tags_assigned} tags assigned | "
            f"Rate: {rate_per_second:.1f}/sec | "
            f"Total: {self.total_processed_assets} processed, {self.total_assigned_tags} tags"
        )
        
        return batch_result
    
    def set_current_library(self, library_name: str):
        """Set the current library for processing."""
        # Initialize failure tracker for this library if not exists
        if library_name not in self.library_failure_trackers:
            self.library_failure_trackers[library_name] = FailureTracker(library_name)
        
        self.failure_tracker = self.library_failure_trackers[library_name]
        
        # Initialize library metrics if not exists
        if library_name not in self.library_metrics:
            self.library_metrics[library_name] = {
                "processed_assets": 0,
                "assigned_tags": 0,
                "failed_assets": 0
            }

        # Immich tags belong to a user. Resolve the marker again whenever the
        # API key changes instead of reusing the first library's tag ID.
        self._initialize_processed_tag()
        self._initialize_content_rating_tags()
    
    def get_unprocessed_assets(self, limit: Optional[int] = None) -> List[Asset]:
        """Get eligible image assets that need processing.

        With no target albums, eligible images have no tags. With target
        albums, eligible images are in a configured album and do not carry the
        processed marker. Videos are excluded because WD14 cannot process them.
        """
        if limit is None:
            limit = settings.batch_size
        
        try:
            # Check for external changes to failure file (e.g., cleanup script ran)
            if self.failure_tracker and self.failure_tracker.check_for_external_changes():
                self.logger.debug("🔄 Failure tracking data refreshed from external changes")
            
            if not self.processed_tag:
                raise ProcessorError("Processed tag has not been initialized")

            assets = self.immich_client.get_unprocessed_assets(
                processed_tag_id=self.processed_tag.id,
                limit=limit
            )
            
            if not assets:
                library_name = self.immich_client.current_library_name
                self.logger.info(f"✅ Library '{library_name}': No more eligible images found - processing complete!")
                return []
            
            # Filter out permanently failed assets
            if self.failure_tracker:
                filtered_assets = self.failure_tracker.filter_failed_assets(assets)
            else:
                filtered_assets = assets
            
            if not filtered_assets:
                if len(assets) > 0:
                    library_name = self.immich_client.current_library_name
                    self.logger.warning(f"⚠️  Library '{library_name}': Found {len(assets)} eligible images, but all are permanently failed!")
                    self.logger.info("💡 Use --show-failures to see failed asset IDs or --reset-failures to retry them")
                return []
            
            library_name = self.immich_client.current_library_name
            self.logger.info(f"🎯 Library '{library_name}': Found {len(filtered_assets)} eligible images to process")
            return filtered_assets
            
        except Exception as e:
            self.logger.error(f"Failed to get unprocessed assets: {str(e)}")
            raise ProcessorError(f"Failed to get unprocessed assets: {e}")
    
    def run_processing_cycle(self) -> bool:
        """Run a single processing cycle."""
        try:
            # Get unprocessed assets
            assets = self.get_unprocessed_assets()
            
            if not assets:
                self.logger.info("✅ All images have been processed!")
                return False
            
            # Process the batch
            batch_result = self.process_batch(assets)
            
            # Return True if there were any assets processed (successful or failed)
            # This indicates we should continue looking for more assets
            assets_processed = batch_result.successful + batch_result.failed
            return assets_processed > 0
            
        except Exception as e:
            self.logger.error(f"❌ Processing cycle failed: {str(e)}")
            raise
    
    def run_continuous_processing(self, max_cycles: Optional[int] = None):
        """Run continuous processing until no more assets are found or max cycles reached."""
        cycle_count = 0
        
        self.logger.info("🚀 Starting continuous processing...")
        
        while True:
            if max_cycles and cycle_count >= max_cycles:
                self.logger.info(f"🔢 Reached maximum cycles: {max_cycles}")
                break
            
            cycle_count += 1
            
            # Run processing cycle
            should_continue = self.run_processing_cycle()
            
            if not should_continue:
                self.logger.info("🎉 Processing complete! No more assets to process.")
                break
            
            # Small delay between cycles to be gentle on the API
            time.sleep(1.0)
        
        # Final summary
        self.logger.info(
            f"🏁 Processing complete! Total: {self.total_processed_assets} assets processed, "
            f"{self.total_assigned_tags} tags assigned in {cycle_count} cycles"
        )
        
        # Log final performance summary
        performance_monitor.log_performance_summary()
    
    def reset_progress(self):
        """Reset processing progress counters."""
        self.total_processed_assets = 0
        self.total_assigned_tags = 0
        self.logger.info("🔄 Progress counters reset")
    
    def get_progress_status(self) -> dict:
        """Get processing progress information."""
        return {
            "total_processed": self.total_processed_assets,
            "total_tags_assigned": self.total_assigned_tags
        }
    
    def get_failure_summary(self) -> dict:
        """Get failure tracking summary."""
        return self.failure_tracker.get_failure_summary()
    
    def get_failed_asset_ids(self, permanently_failed_only: bool = True) -> List[str]:
        """Get list of failed asset IDs.
        
        Args:
            permanently_failed_only: If True, only return permanently failed assets.
                                   If False, return all failed assets (including retry candidates).
        """
        if permanently_failed_only:
            return self.failure_tracker.get_permanently_failed_assets()
        else:
            return list(self.failure_tracker.get_failed_assets().keys())
    
    def reset_failures(self, asset_ids: List[str] = None):
        """Reset failure tracking for specific assets or all assets.
        
        Args:
            asset_ids: List of asset IDs to reset, or None to reset all failures
        """
        self.failure_tracker.reset_failures(asset_ids)

    def get_metrics(self):
        """Get current processing metrics."""
        base_metrics = self.metrics.get_metrics()
        performance_metrics = performance_monitor.get_metrics_dict()
        progress_info = self.get_progress_status()
        
        # Combine all metric sources
        combined_metrics = {
            "basic_metrics": base_metrics,
            "performance_metrics": performance_metrics,
            "progress_info": progress_info
        }
        
        return combined_metrics
    
    def test_connection(self) -> bool:
        """Test the connection to Immich."""
        return self.immich_client.test_connection()
    
    def close(self):
        """Clean up resources."""
        self.immich_client.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
