#!/usr/bin/env bash
# Publishes the documentation to the gh-pages branch with mike, and prunes the
# release versions that no longer fit on GitHub Pages.
#
#   publish-docs.sh dev                  master's docs as `dev`; the site's
#                                        default while no release is `latest`
#   publish-docs.sh latest <version>     a release, which `latest` moves to
#   publish-docs.sh older <version>      an older release; `latest` stays
#
# Why prune: every release used to stay on gh-pages for good, about 24 MB
# each. At about 490 versions the branch held 11 GB, and the Pages deployment
# refused it ("Artifact could not be deployed ... total size is less than
# 10GB"; Pages sites are limited to 1 GB). Only `dev` and the newest
# KEEP_RELEASES releases stay online, as well as whatever version holds an
# alias; every release keeps its full documentation as the .zip and .pdf
# assets of its GitHub Release.
#
# Why retry: the Docs and Release workflows both run on a merge to master and
# both push to gh-pages, so one push can be rejected as non-fast-forward. A
# rejected attempt starts over from the new gh-pages tip.
#
# Environment:
#   CONFIG          the mkdocs config (default autoware_carla_scenario/mkdocs.yml)
#   KEEP_RELEASES   how many releases stay online (default 20)
set -euo pipefail

MODE="${1:?usage: publish-docs.sh dev|latest|older [version]}"
VERSION="${2:-}"
CONFIG="${CONFIG:-autoware_carla_scenario/mkdocs.yml}"
KEEP_RELEASES="${KEEP_RELEASES:-20}"
ATTEMPTS=5

case "${MODE}" in
  dev) ;;
  latest | older)
    if ! [[ "${VERSION}" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
      echo "::error::'${VERSION}' is not a version like 3.1.2"
      exit 1
    fi
    ;;
  *)
    echo "::error::unknown mode '${MODE}' (dev, latest or older)"
    exit 1
    ;;
esac
if ! [[ "${KEEP_RELEASES}" =~ ^[1-9][0-9]*$ ]]; then
  echo "::error::KEEP_RELEASES must be a positive number, not '${KEEP_RELEASES}'"
  exit 1
fi

mike() { uv run mike "$@" --config-file "${CONFIG}"; }

# Makes the local gh-pages the remote one, dropping a rejected attempt.
sync_gh_pages() {
  if git fetch --quiet --depth=1 origin \
      "+refs/heads/gh-pages:refs/remotes/origin/gh-pages" 2>/dev/null; then
    git update-ref refs/heads/gh-pages refs/remotes/origin/gh-pages
  fi
}

publish() {
  case "${MODE}" in
    latest)
      mike deploy --update-aliases "${VERSION}" latest
      mike set-default latest
      ;;
    older)
      mike deploy "${VERSION}"
      ;;
    dev)
      mike deploy --title "dev (master)" dev
      # Point the site root at dev only while there is no release to point at.
      if ! mike list --json | jq -e 'any(.[]; .aliases | index("latest"))' >/dev/null; then
        mike set-default dev
      fi
      ;;
  esac
}

# Deletes the releases older than the newest KEEP_RELEASES. A version that
# holds an alias (`latest`) and anything not shaped X.Y.Z (`dev`) is kept.
prune() {
  local stale
  stale=$(mike list --json \
    | jq -r '.[] | select((.aliases | length) == 0) | .version
                 | select(test("^[0-9]+\\.[0-9]+\\.[0-9]+$"))' \
    | sort -V -r \
    | tail -n "+$((KEEP_RELEASES + 1))")
  if [ -z "${stale}" ]; then
    echo "No release docs to prune (keeping the newest ${KEEP_RELEASES})."
    return
  fi
  echo "Pruning $(echo "${stale}" | wc -l) release docs older than the newest ${KEEP_RELEASES}:"
  echo "${stale}" | tr '\n' ' '
  echo
  # shellcheck disable=SC2086 # one version per word
  mike delete -m "Prune docs older than the newest ${KEEP_RELEASES} releases" ${stale}
}

for attempt in $(seq 1 "${ATTEMPTS}"); do
  sync_gh_pages
  publish
  prune
  if git push origin gh-pages; then
    exit 0
  fi
  echo "::warning::gh-pages moved during attempt ${attempt}; retrying from its new tip"
  sleep $((attempt * 10))
done
echo "::error::could not push gh-pages after ${ATTEMPTS} attempts"
exit 1
