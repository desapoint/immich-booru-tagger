#!/bin/sh
set -eu

PUID="${PUID:-99}"
PGID="${PGID:-100}"
UMASK="${UMASK:-022}"

echo "----------------------------------------------------"
echo " Immich Booru Tagger"
echo "----------------------------------------------------"
echo " PUID:  ${PUID}"
echo " PGID:  ${PGID}"
echo " UMASK: ${UMASK}"
echo "----------------------------------------------------"

umask "${UMASK}"

# Persistent runtime directories.
mkdir -p \
    /config \
    /config/models \
    /config/cache \
    /config/huggingface \
    /config/torch

#
# PUID 0 explicitly means run as root.
#
if [ "${PUID}" = "0" ]; then
    cd /config
    exec "$@"
fi

#
# Find or create the requested group.
#
if getent group "${PGID}" >/dev/null 2>&1; then
    APP_GROUP="$(getent group "${PGID}" | cut -d: -f1)"
else
    APP_GROUP="booru"
    groupadd -g "${PGID}" "${APP_GROUP}"
fi

#
# Find or create the requested user.
#
if getent passwd "${PUID}" >/dev/null 2>&1; then
    APP_USER="$(getent passwd "${PUID}" | cut -d: -f1)"

    CURRENT_GID="$(id -g "${APP_USER}")"

    if [ "${CURRENT_GID}" != "${PGID}" ]; then
        usermod -g "${APP_GROUP}" "${APP_USER}"
    fi
else
    APP_USER="booru"

    useradd \
        -u "${PUID}" \
        -g "${APP_GROUP}" \
        -d /config \
        -s /usr/sbin/nologin \
        "${APP_USER}"
fi

#
# Fix ownership of persistent data automatically.
#
chown -R "${PUID}:${PGID}" /config

#
# FailureTracker writes relative filenames such as:
#
# processing_failures_Library_1.json
#
# Running from /config makes them persistent.
#
cd /config

echo "Starting Immich Booru Tagger as ${PUID}:${PGID}"

exec gosu "${PUID}:${PGID}" "$@"