# CARLA Maps from Content Packs

A CARLA map the server does not ship -- a town cooked as a **content pack** --
is named by the pack that carries it. The framework downloads the pack, reads
its HD map, mounts it on the server and runs the scenario on it, all through
[carla-pack-registry](https://github.com/autowarefoundation/carla_content_pack_registry):

```bash
uv run scenario scenario=my_scenario/default \
  map.name=NishishinjukuMap map.pack=tier4/carla-packs/NishishinjukuPack@1.2.0
```

Packs are shared through GitHub repository permissions: a pack repository
lists its packs in `carla-packs.toml`, and whoever can read that repository
can download them (straight from S3-compatible storage, or from a public
Hugging Face repository). A pack repository can also keep its packs on a
Google Drive shared drive: then the drive's sharing decides, too, and each
user reads it with their own Google account through
[rclone](https://rclone.org/). There is no server to run. How to publish packs is
described in carla-pack-registry's own documentation.

## Setting it up

Install the `pack` extra, and give carla-pack a GitHub token that can read the
pack repository:

```bash
pip install 'autoware-carla-scenario[pack]'   # or: uv add carla-pack-registry

gh auth login                     # carla-pack uses the GitHub CLI's token as is
carla-pack whoami                 # who carla-pack reads the repositories as
carla-pack ls tier4/carla-packs   # the packs you can download
```

If the pack repository keeps its packs on Google Drive (`carla-pack
credentials OWNER/REPO` says "Google Drive"), also install rclone (v1.64 or
later) and sign in with the Google account the drive is shared with, once:

```bash
carla-pack drive login                     # opens the browser; --headless on a server
carla-pack credentials tier4/carla-packs   # checks that the drive can be read
```

In CI, share the drive with a Google service account and set
`CARLA_PACK_RCLONE_SERVICE_ACCOUNT_FILE` to its JSON key instead. A run in a
container needs rclone and those credentials inside it.

The token is looked up as carla-pack looks it up: `CARLA_PACK_TOKEN`, then
`GH_TOKEN`, `GITHUB_TOKEN`, the one `carla-pack login` saved, and `gh auth
token`. In CI, pass a fine-grained token that can read the pack repository in
`CARLA_PACK_TOKEN` (a workflow's own `GITHUB_TOKEN` cannot read another
repository).

## Naming the pack

`map.pack` takes one of two forms:

| `map.pack` | What is run |
| --- | --- |
| `OWNER/REPO/PACK@VERSION` | That version of the pack |
| `OWNER/REPO/PACK` | The newest version of the pack |
| `OWNER/REPO` | The newest pack in the repository providing `map.name` |

`map.pack_sha256` pins the pack to one artifact, as a lock file would. Without
it, the run still mounts exactly the artifact its map files were read from:
the reference is resolved once, before the run, and its SHA-256 carried on.

A map group can name it once for every scenario:

```yaml
# conf/map/nishishinjuku_pack.yaml
# @package _global_
map:
  name: NishishinjukuMap
  pack: tier4/carla-packs/NishishinjukuPack@1.2.0
```

## What happens

1. **Before CARLA starts**, the pack is downloaded into carla-pack's cache
   (`~/.cache/carla-pack/blobs/<sha256>.tar.gz`, `CARLA_PACK_CACHE_DIR`),
   verified against the repository manifest's SHA-256 and the pack's own
   `carla-pack.json`, and unpacked there. Its map data is read by format:
   - the **Lanelet2 map** from the pack's `autoware` folder (the Lanelet2
     map, `map_projector_info.yaml` and the point cloud map), or else its bare
     `lanelet2` map -- read as [`map.directory`](maps.md) reads a directory;
   - the **OpenDRIVE** CARLA cooks the map from (`opendrive`), which becomes
     the road network the run reads, so nothing has to be read back out of
     the loaded world.

   A pack that carries no Lanelet2 map is still mounted; name the Lanelet2 map
   with `map.directory`, `map.source` or `map.lanelet2_path` then.
2. **Once the server is up** -- the one the run starts, or one already
   running -- the pack is installed into `server.packs_dir` and mounted on the
   server, before the map is loaded. A pack already installed is not
   downloaded again (the install is made from the same cached tarball), and
   one already mounted is left as it is. A server the queue restarts after it
   died mounts it again.
3. The map is loaded by `map.name` and the scenario runs as on any other map.

With `map.pack=OWNER/REPO`, a server that already has `map.name` (built in,
or mounted earlier) is used as it is, and nothing is mounted.

## Precedence

| Setting | Wins over |
| --- | --- |
| `map.lanelet2_path`, `map.xodr_path` | everything below |
| `map.directory` | the pack's Lanelet2 map |
| `map.pack` | `map.source` |
| `map.source` | -- |

`map.directory` wins over the pack's Lanelet2 map only: the pack is still
mounted, and its OpenDRIVE is still read unless the directory ships one. A
batch's cases must agree on `map.pack` and `map.pack_sha256`, as they must on
every other map setting.

The trajectory log records the map's source as the pack the run read,
`carla-pack:OWNER/REPO/PACK@VERSION` -- the version included, also for
`map.pack=OWNER/REPO` or a pack named without one. When the pack carries no
Lanelet2 map and `map.source` gives it, the log names `map.source` instead.
`map.pack_sha256` without `map.pack` is refused: there is nothing to pin.

## Where packs are installed

The server reads a pack at the path it is installed at:

| Setting | Default |
| --- | --- |
| `server.packs_dir` | `null` |
| `AUTOWARE_CARLA_SCENARIO_PACKS_DIR` | -- |
| otherwise | `~/.autoware_carla_scenario/packs` (`AUTOWARE_CARLA_SCENARIO_HOME/packs`) |

A relative path is taken from the working directory of the run and made
absolute before it is handed to the server. A blank `server.packs_dir` is
refused; `null` is the default.

A server on another machine (`server.host`) needs a directory it reads at the
same path as this machine -- a shared mount -- or the mount is refused. Each
installed pack takes its own size on disk, beside the cached tarball.

A multi-gigabyte pack is downloaded once; a dropped connection resumes where
it stopped.

## From Python

```python
from autoware_carla_scenario import ScenarioQueue
from autoware_carla_scenario.maps import resolve_map_paths
from autoware_carla_scenario.maps.pack import MapPack

# Downloads and verifies the pack, and reads its Lanelet2 map and OpenDRIVE.
paths = resolve_map_paths(
    {"name": "NishishinjukuMap", "pack": "tier4/carla-packs/NishishinjukuPack@1.2.0"}
)

queue = ScenarioQueue(
    map_name=paths.name,
    xodr_path=paths.xodr_path,
    lanelet2_path=paths.lanelet2_path,
    map_pack=MapPack(paths.pack, sha256=paths.pack_sha256),
)
queue.add(MyScenario(ego_config))
with queue:  # starts CARLA, mounts the pack, loads the map
    results = queue.run_all()
```

`MapPack.files(map_name)` alone hands back the pack's map files
(`directory`, `xodr_path`, and the resolved `reference` and `sha256`), and
`MapPack.deploy(host, port, map_name)` alone mounts the pack on a server.

## Troubleshooting

When GitHub or the pack's storage refuses access, the run stops with a
`PackAccessError` (a `MapPackError`) that says what was refused, where the
token was taken from, and what to do. Its `code` is carla-pack's:

| Message | `code` | Cause and fix |
| --- | --- | --- |
| `GitHub authentication failed. No GitHub token was found ...` | `unauthenticated` | No token in `CARLA_PACK_TOKEN`, `GH_TOKEN`, `GITHUB_TOKEN`, `carla-pack login` or `gh auth token`, and the repository is private: run `gh auth login`, or set `CARLA_PACK_TOKEN` |
| `GitHub authentication failed. GitHub rejected the token from <source> ...` | `unauthenticated` | The token has expired or been revoked. The message names where it came from and how to renew it (`gh auth login`, `carla-pack login`, or the variable to set). A GitHub Actions workflow's own `GITHUB_TOKEN` cannot read another repository |
| `GitHub authentication failed. GitHub's limit on requests without a token ...` | `unauthenticated` | The anonymous rate limit: give carla-pack a token |
| `access denied. OWNER/REPO does not exist, or the GitHub user <login> ... cannot read it` | `no_access` | The token works but its user has no Read access to the pack repository (GitHub answers a private repository with *not found*): ask its owner for access, or use another user's token in `CARLA_PACK_TOKEN` |
| `the storage credentials of OWNER/REPO could not be read. Your GitHub access is fine ...` | `credentials_unavailable` | The pack repository's read-only storage key expired: its credential workflow has to run (the owner's to fix) |
| `the pack's storage refused the download ...` | `storage_error` | The storage key was rotated while downloading: retry |
| `... keeps its packs on Google Drive, which carla-pack reads through rclone, and rclone cannot run here` | `rclone_unavailable` | rclone is not installed (or older than v1.64): install it, or set `CARLA_PACK_RCLONE` to its path |
| `Google Drive authentication failed ...` | `storage_unauthenticated` | No rclone remote for your Google account, or Google rejected its token: run `carla-pack drive login` |
| `Google Drive access denied ...` | `storage_no_access` | GitHub let you read the pack repository, but its shared drive is not shared with your Google account (Google Drive answers that like a missing file): ask the owner to share it |

`carla-pack whoami` shows which user and token carla-pack uses, and
`carla-pack ls OWNER/REPO` what it can read.

Other errors:

| Message | Cause |
| --- | --- |
| `Install carla-pack-registry to use map.pack` | The `pack` extra is not installed |
| `Could not find the pack ...: ... not found` | The pack or version does not exist in a repository you can read |
| `The pack ... does not carry the map ...` | `map.name` is not one of the pack's maps; the message lists them |
| `... did not mount the pack ...` | The server cannot read `server.packs_dir` at that path |
| `Could not deploy the pack ...` with *incompatible* | The pack was built for another CARLA release than the server's |
