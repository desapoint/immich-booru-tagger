# Immich Booru-Tagger

AI-powered image tagging for Immich using the WD SwinV2 v3 anime model. It runs the model through ONNX Runtime and automatically applies booru-style tags to your anime/manga images.

## Quick Start

### Docker (Recommended)
```bash
git clone https://github.com/desapoint/immich-booru-tagger.git
cd immich-booru-tagger
cp .env.example .env
# Edit .env with your Immich settings
docker-compose up -d
```

### Python
```bash
pip install -r requirements.txt
export IMMICH_BASE_URL="https://your-immich-server.com"
export IMMICH_API_KEY="your-api-key"
python -m immich_tagger.main
```

## How It Works

1. **Finds Images**: Finds untagged images globally, or untreated images in configured target albums
2. **AI Processing**: Downloads up to `BATCH_SIZE` assets and evaluates them in bounded ONNX batches
3. **Auto-Tagging**: Applies predicted tags with confidence filtering
4. **Self-Managing**: A processed marker excludes completed images from future runs
5. **Repeats**: Continues until no eligible images remain or the configured per-run batch limit is reached

**Features**: Resumable, batched inference, self-managing, persistent model cache, multi-library support.

## Configuration

### Essential Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `IMMICH_BASE_URL` | Your Immich server URL | Required |
| `IMMICH_API_KEY` | API key (single library) | Required |
| `IMMICH_API_KEYS` | Multiple API keys (JSON array) | `[]` |
| `CONFIDENCE_THRESHOLD` | Minimum tag confidence | `0.35` |
| `CHARACTER_THRESHOLD` | Minimum character-tag confidence | `0.9` |
| `BATCH_SIZE` | Assets downloaded per processing cycle | `250` |
| `MAX_BATCHES_PER_RUN` | Maximum batches across all libraries in one scheduled run (`0` = unlimited) | `0` |
| `INFERENCE_BATCH_SIZE` | Images evaluated in one ONNX call | `8` |
| `DOWNLOAD_WORKERS` | Concurrent thumbnail downloads within each inference chunk | `4` |
| `PROCESSED_TAG_NAME` | Marker used to prevent reprocessing | `auto:processed` |
| `CONTENT_RATING_TAG_NAME` | Parent for hierarchical content ratings | `content-rating` |
| `TARGET_ALBUMS` | Comma-separated album names to process | Empty |
| `FAILURE_TIMEOUT` | Max retries for failed assets | `3` |

Optional model tuning:

| Variable | Description | Default |
|----------|-------------|---------|
| `WD_MODEL_REPO` | Hugging Face repository containing `model.onnx` and `selected_tags.csv` | `SmilingWolf/wd-swinv2-tagger-v3` |
| `MODEL_CACHE_DIR` | Downloaded model cache path | `/config/models` |
| `ONNX_INTRA_OP_THREADS` | ONNX CPU worker threads (`0` lets the runtime decide) | `0` |
| `UNLOAD_MODEL_AFTER_RUN` | Release model memory between scheduled runs, except when the next run is under 15 minutes away | `false` |

You do not need to add these optional variables to an existing Unraid Compose
file. The image defaults are suitable for normal use. Start with
`INFERENCE_BATCH_SIZE=8`; lower it if memory is constrained, or raise it
gradually if the host has spare RAM and CPU capacity. `DOWNLOAD_WORKERS=4`
keeps network concurrency conservative; raise it only if Immich and the network
have spare capacity.

When `UNLOAD_MODEL_AFTER_RUN=true`, the scheduler releases the ONNX session
after a complete multi-library run if the next cron occurrence is at least 15
minutes away. It keeps the session resident when the next run is less than 15
minutes away. Model loading, retention decisions, and unload completion are
logged at `INFO`; downloaded files remain cached under `/config/models`.

`MAX_BATCHES_PER_RUN` bounds the work performed by each scheduler occurrence.
For example, `BATCH_SIZE=250` and `MAX_BATCHES_PER_RUN=4` process at most
1,000 assets per run. The allowance is shared across configured libraries;
the scheduler processes one batch per library in round-robin passes and rotates
the starting library between capped runs. Remaining eligible images resume at
the next scheduled run. Runs are guarded so a second trigger is skipped while
an earlier run is still active.

The run guard combines an in-process lock with an advisory lock at
`/config/.processing-run.lock`. This prevents overlapping scheduler containers
or manually launched `single`/`continuous` sessions when they share the same
`/config` mount. The file may remain after a crash, but ownership is tied to an
open file descriptor and is automatically released when that process exits.

### Verifying the running image

Every published image reports its release identity at startup and through
`/`, `/health`, and `/metrics`. The `revision` value is the exact Git commit
used by GitHub Actions. For example:

```bash
curl http://your-unraid-host:8000/health
docker inspect immich_booru_tagger --format '{{ index .Config.Labels "org.opencontainers.image.revision" }}'
```

The revision in both outputs should match. Local source builds report
`version=development` and `revision=unknown` unless build arguments are passed.

### Content ratings

The tagger creates a dedicated hierarchy for WD14's four content ratings:

```text
content-rating
├── general
├── sensitive
├── questionable
└── explicit
```

New predictions are assigned to the child tags rather than flat rating tags.
On startup, this hierarchy is ensured separately for every configured Immich
user. If a flat `general`, `sensitive`, `questionable`, or `explicit` tag
already exists, its asset associations are copied to the matching child before
the flat tag is removed. Ratings that have no legacy flat tag skip association
migration entirely.

Immich does not currently allow a tag to be renamed or reparented through its
update endpoint, so migration uses create, assign, unassign, and delete
operations. If migration is interrupted, the source tag is retained and the
operation is safe to retry at the next launch.

### Target Albums

Leave `TARGET_ALBUMS` empty to preserve the original behavior, which processes
images that have no tags at all. To limit processing to one or more albums, use
a comma-separated list:

```bash
TARGET_ALBUMS=Anime,Hentai
```

Album names are matched case-insensitively. Assets in any matching album are
processed unless they already carry `PROCESSED_TAG_NAME`. An invalid album name
is logged and skipped while valid configured albums continue. If none of the
configured names exist, that library run fails.

`TARGET_ALBUMS` applies to every configured Immich library/user, so each one
must have at least one matching accessible album.

### Multi-Library Support

```bash
# Multiple users
IMMICH_API_KEYS='["key1", "key2"]'

# Named libraries
IMMICH_LIBRARIES='{"Alice": "key1", "Bob": "key2"}'
```

## Usage

### Processing Modes

```bash
# Test connection
python -m immich_tagger.main --test-connection

# Process one batch
python -m immich_tagger.main --mode single

# Continuous processing (recommended for bulk)
python -m immich_tagger.main --mode continuous

# Scheduled processing (daily at 2 AM)
python -m immich_tagger.main --mode scheduler

# Health monitoring only
python -m immich_tagger.main --mode health-only
```

### Failure Management

```bash
# View failed assets
python -m immich_tagger.main --show-failures

# Reset all failures
python -m immich_tagger.main --reset-failures

# Clean up permanently failed assets
python cleanup_failed_assets.py --dry-run  # Preview
python cleanup_failed_assets.py            # Remove
python cleanup_failed_assets.py --force    # Force removal
```

## Health Monitoring

- **Health Check**: `http://localhost:8000/health`
- **Metrics**: `http://localhost:8000/metrics`
- **Service Info**: `http://localhost:8000/`

AI processing runs in a worker thread, so a large inference batch does not
block the asynchronous health listener. The first container start still needs
time to download and initialize the model; the supplied Docker health check
allows a five-minute startup period.

## AI Model

- **WD SwinV2 v3**: Anime-optimized booru tagging via the official
  `SmilingWolf/wd-swinv2-tagger-v3` ONNX export.
- Inference preserves the model's raw label names and category thresholds.
  Batched ONNX changes how images are evaluated, not which label vocabulary is
  used.
- PyTorch and TensorFlow are not installed or required.

## Performance

- **Efficiency**: Dynamic ONNX batches avoid one model invocation per image
- **Memory control**: Asset-fetch and inference batch sizes are independently configurable
- **Resumable**: Always picks up where it left off
- **Multi-Library**: Processes all libraries sequentially

## Troubleshooting

### Common Issues

1. **Connection Failed**: Check `IMMICH_BASE_URL` and API key scopes
2. **No Assets Processed**: Verify assets don't already have `auto:processed` tag
3. **Model Issues**: Ensure `/config` is writable and has enough space for the model cache

### API Key Requirements

Your Immich API key needs these scopes:
- `asset.read` - List and search assets
- `asset.view` / `asset.download` - Download thumbnails or originals
- `asset.update` - Allow tag associations to update assets
- `tag.read` - Find existing tags
- `tag.create` - Create predicted and processed-marker tags
- `tag.asset` - Assign tags to assets
- `tag.delete` - Remove migrated legacy flat content-rating tags
- `album.read` - Resolve and search `TARGET_ALBUMS` when album filtering is enabled

## Architecture

```
Immich API ←→ Auto-Tagger ←→ WD SwinV2 v3 (ONNX Runtime)
                ↓
         Health Server (Port 8000)
```

## Persistent data

Mount one persistent directory at `/config`. It contains the Hugging Face/model
cache and per-library failure records:

```yaml
volumes:
  - ./config:/config
```

If upgrading from the old repository Compose layout, replace the separate
`/app/models` and `/app/processing_failures.json` mounts with this one. Existing
failure JSON files can be copied into `/config` using their current
`processing_failures_<library>.json` names; the model will otherwise download
once into the new cache.
