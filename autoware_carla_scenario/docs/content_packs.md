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
Hugging Face repository). There is no server to run. How to publish packs is
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

The trajectory log records the map's source as `carla-pack:OWNER/REPO/PACK[@VERSION]`.

## Where packs are installed

The server reads a pack at the path it is installed at:

| Setting | Default |
| --- | --- |
| `server.packs_dir` | `null` |
| `AUTOWARE_CARLA_SCENARIO_PACKS_DIR` | -- |
| otherwise | `~/.autoware_carla_scenario/packs` (`AUTOWARE_CARLA_SCENARIO_HOME/packs`) |

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

| Message | Cause |
| --- | --- |
| `Install carla-pack-registry to use map.pack` | The `pack` extra is not installed |
| `Could not find the pack ...: ... gh auth login ...` | GitHub does not let the token read the repository (a private repository answers *not found*), or the pack or version does not exist |
| `The pack ... does not carry the map ...` | `map.name` is not one of the pack's maps; the message lists them |
| `... did not mount the pack ...` | The server cannot read `server.packs_dir` at that path |
| `Could not deploy the pack ...` with *incompatible* | The pack was built for another CARLA release than the server's |
