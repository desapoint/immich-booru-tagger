"""
Scheduler for continuous operation of the Immich Auto-Tagger.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Optional
from croniter import croniter
import pytz

from .config import settings
from .logging import get_logger
from .processor import ImmichAutoTagger
from .run_lock import ProcessingRunLock


MODEL_KEEP_LOADED_WINDOW = timedelta(minutes=15)


class Scheduler:
    """Scheduler for running the auto-tagger at specified intervals."""
    
    def __init__(self, processor: Optional[ImmichAutoTagger] = None):
        self.logger = get_logger("scheduler")
        self.processor = processor or ImmichAutoTagger()
        self.running = False
        self.timezone = pytz.timezone(settings.timezone)
        self.last_run_time: Optional[datetime] = None
        self._processing_lock = ProcessingRunLock(purpose="scheduler")
        
    def _get_next_run_time(self) -> datetime:
        """Get the next scheduled run time based on cron expression."""
        now = datetime.now(self.timezone)
        cron = croniter(settings.cron_schedule, now)
        return cron.get_next(datetime)
    
    def _should_run_now(self) -> bool:
        """Check if it's time to run based on the cron schedule."""
        now = datetime.now(self.timezone)
        
        # If we've never run, check if we're past the first scheduled time
        if self.last_run_time is None:
            # Get the most recent scheduled time (previous occurrence)
            cron = croniter(settings.cron_schedule, now)
            last_scheduled = cron.get_prev(datetime)
            
            # If the last scheduled time was within the last 24 hours, we should run
            time_since_scheduled = (now - last_scheduled).total_seconds()
            return time_since_scheduled <= 86400  # 24 hours
        
        # If we have run before, check if there's been a scheduled time since our last run
        cron = croniter(settings.cron_schedule, self.last_run_time)
        next_after_last_run = cron.get_next(datetime)
        
        # If the next scheduled time after our last run is now or in the past, we should run
        return now >= next_after_last_run
    
    async def _run_processing_cycle(self):
        """Run synchronous multi-library processing outside the event loop."""
        return await asyncio.to_thread(self._run_processing_cycle_sync)

    def _manage_model_retention(self):
        """Unload the model unless another scheduled run is less than 15m away."""
        if (
            not settings.unload_model_after_run
            or not self.processor.is_tagging_engine_loaded
        ):
            return

        try:
            now = datetime.now(self.timezone)
            next_run = self._get_next_run_time()
            time_until_next_run = next_run - now
            minutes_until_next_run = max(
                0.0,
                time_until_next_run.total_seconds() / 60,
            )

            if timedelta(0) <= time_until_next_run < MODEL_KEEP_LOADED_WINDOW:
                self.logger.info(
                    "🧠 Keeping ONNX model loaded: next run is in "
                    f"{minutes_until_next_run:.1f} minutes (< 15 minutes)"
                )
                return

            self.logger.info(
                "💤 Next run is in "
                f"{minutes_until_next_run:.1f} minutes; unloading ONNX model"
            )
            self.processor.unload_tagging_engine()
        except Exception as e:
            self.logger.error(f"❌ Failed to manage ONNX model retention: {e}")

    def _run_processing_cycle_sync(self):
        """Run a processing cycle for all libraries."""
        if not self._processing_lock.acquire(blocking=False):
            owner_description = getattr(
                self._processing_lock,
                "owner_description",
                lambda: "owner details unavailable",
            )()
            self.logger.warning(
                "⏭️  Processing run skipped: another run is already active "
                f"({owner_description})"
            )
            return False

        try:
            self.logger.info("🚀 Starting scheduled multi-library processing cycle")
            
            # Update last run time
            self.last_run_time = datetime.now(self.timezone)
            
            total_processed = 0
            total_tags = 0
            batches_processed = 0
            batch_limit_reached = False
            library_configs = self.processor.immich_client.library_configs
            
            for i, library_config in enumerate(library_configs):
                library_name = library_config["name"]
                
                try:
                    # Get user info for this library
                    self.processor.immich_client.switch_to_library(i)
                    user_info = self.processor.immich_client.get_current_user_info()
                    
                    self.logger.info(f"🏛️ Processing library '{library_name}' ({i+1}/{len(library_configs)}) - User: {user_info['name']} ({user_info['email']})")
                    
                    # Set current library in processor
                    self.processor.set_current_library(library_name)
                    
                    # Process this library until complete or this run reaches
                    # its global batch allowance.
                    library_start_processed = self.processor.library_metrics.get(library_name, {}).get("processed_assets", 0)
                    library_start_tags = self.processor.library_metrics.get(library_name, {}).get("assigned_tags", 0)
                    library_complete = False
                    
                    while True:
                        if (
                            settings.max_batches_per_run > 0
                            and batches_processed >= settings.max_batches_per_run
                        ):
                            batch_limit_reached = True
                            break

                        cycle_result = self.processor.run_processing_cycle()
                        if not cycle_result:
                            library_complete = True
                            break
                        batches_processed += 1
                    
                    # Calculate library totals
                    library_processed = self.processor.library_metrics.get(library_name, {}).get("processed_assets", 0) - library_start_processed
                    library_tags = self.processor.library_metrics.get(library_name, {}).get("assigned_tags", 0) - library_start_tags
                    
                    total_processed += library_processed
                    total_tags += library_tags
                    
                    if library_complete:
                        self.logger.info(f"✅ Library '{library_name}' complete: {library_processed} assets processed, {library_tags} tags assigned")
                    else:
                        self.logger.info(
                            f"⏸️  Library '{library_name}' paused: "
                            f"{library_processed} assets processed, "
                            f"{library_tags} tags assigned"
                        )

                    if batch_limit_reached:
                        break
                    
                except Exception as e:
                    self.logger.error(f"❌ Error processing library '{library_name}': {e}")
                    continue
            
            if batch_limit_reached:
                self.logger.info(
                    f"🔢 Reached per-run limit of "
                    f"{settings.max_batches_per_run} batches; remaining "
                    "eligible images will wait until the next run"
                )
                self.logger.info(
                    f"🏁 Run complete: {batches_processed} batches, "
                    f"{total_processed} total assets, {total_tags} total tags assigned"
                )
            else:
                self.logger.info(f"🎉 All libraries processed: {total_processed} total assets, {total_tags} total tags assigned")

            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error during scheduled processing cycle: {e}")
            return False
        finally:
            try:
                self._manage_model_retention()
            finally:
                self._processing_lock.release()
    
    async def _scheduler_loop(self):
        """Main scheduler loop."""
        self.logger.info(f"Starting scheduler - Schedule: {settings.cron_schedule}, Timezone: {settings.timezone}")
        
        while self.running:
            try:
                should_run = self._should_run_now()
                self.logger.debug(f"🔍 Should run now: {should_run}")
                
                if should_run:
                    await self._run_processing_cycle()
                    
                    # Calculate next run time
                    next_run = self._get_next_run_time()
                    self.logger.info(f"⏭️  Next scheduled run: {next_run.isoformat()}")
                
                # Sleep for a minute before checking again
                await asyncio.sleep(60)
                
            except Exception as e:
                self.logger.error(f"Error in scheduler loop: {e}")
                await asyncio.sleep(60)  # Wait before retrying
    
    async def start(self):
        """Start the scheduler."""
        if not settings.enable_scheduler:
            self.logger.info("Scheduler disabled, running single continuous processing session")
            await self._run_processing_cycle()
            return
        
        self.running = True
        
        # Show initial schedule
        next_run = self._get_next_run_time()
        self.logger.info(f"⏰ Scheduler started - Schedule: {settings.cron_schedule}, Timezone: {settings.timezone}, Next run: {next_run.isoformat()}")
        
        # Check if we should run immediately (first time or missed schedule)
        if self._should_run_now():
            self.logger.info("🚀 Running immediately (first time or missed schedule)")
            await self._run_processing_cycle()
        
        await self._scheduler_loop()
    
    def stop(self):
        """Stop the scheduler."""
        self.logger.info("Stopping scheduler")
        self.running = False


async def run_scheduler():
    """Run the scheduler."""
    scheduler = Scheduler()
    try:
        await scheduler.start()
    except KeyboardInterrupt:
        scheduler.stop()
        print("\nScheduler stopped by user")


if __name__ == "__main__":
    asyncio.run(run_scheduler())
