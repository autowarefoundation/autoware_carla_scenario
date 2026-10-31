# HD Maps from a Git Repository

An HD map is not a file that belongs next to a scenario. It is published, it is
versioned, and several scenarios share it. So a scenario names its map with a
**source URI** pointing at a git repository, and the framework fetches it,
caches it, and reads it from there:

```yaml
map:
  name: Town10HD_Opt
  source: git+https://huggingface.co/datasets/AutowareFoundation/carla-ue5-maps@dea2cfeab0e00238c3bdbefc48621cefb3dc04a2#autoware_maps/Town10HD_Opt
```

Everything below is available two ways, because a scenario is written two ways:
in the [Scenario Editor](scenario_editor.md), and in the library that runs one.

## The URI

```
git+https://huggingface.co/datasets/OWNER/NAME@REF#PATH
\________________ repository ______________/ \_ref_/ \_path_/
```

Only the repository is required. `@REF` selects a branch, tag or commit and
defaults to the repository's own default branch; `#PATH` selects the directory
holding the map and defaults to the repository root.

How the string is split:

1. A leading `git+` is dropped.
2. `#PATH` is everything after the first `#`.
3. `@REF` is everything after the **last `@` of the repository path**. The
   repository path starts after the host part — after `scheme://host/` for a
   URL, after `user@host:` for an scp-like URL — so an `@` in the host part is
   the user name, never a ref. A ref may therefore contain `/`:

| URI | Repository | Ref | Path |
| --- | --- | --- | --- |
| `git+https://github.com/o/r@feature/x#maps` | `https://github.com/o/r` | `feature/x` | `maps` |
| `https://user@example.com/o/r@v1` | `https://user@example.com/o/r` | `v1` | |
| `git@github.com:o/r.git@release/1.0#maps/T` | `git@github.com:o/r.git` | `release/1.0` | `maps/T` |
| `git@example.com:maps.git@v1` | `git@example.com:maps.git` | `v1` | |
| `file:///srv/maps@main#T` | `file:///srv/maps` | `main` | `T` |

Because the last `@` decides, a ref may not contain an `@`, and neither may
the path of a remote repository (write one as `%40`:
`https://example.com/o/a%40b@v1`).

A local repository — a plain absolute path or a `file://` URL — may have an
`@` in a directory name (a Nix store path, a `user@domain` home directory), so
there the filesystem decides:

1. a path that is an existing directory as a whole has no ref:
   `/srv/maps@2/repo` is the repository `/srv/maps@2/repo`;
2. otherwise the ref follows the last `@` whose left-hand side is an existing
   directory: `/srv/maps@2/repo@feature/x` is the repository
   `/srv/maps@2/repo` at `feature/x`;
3. if no such `@` exists on this machine, the last `@` decides, as above.

To take the filesystem out of it, write the `@` of a directory name as `%40`
in a `file://` URL (`file:///srv/maps%402/repo@main`); git decodes it.

The URL you actually have is the one in your browser's address bar, and that is
accepted as-is and normalised — GitHub's `/tree/<ref>/<path>` and HuggingFace's
`/tree/`, `/resolve/` and `/raw/` forms all parse. Paste the page you are
looking at.

Any git server works: HuggingFace, GitHub, GitLab, an internal host, a
`file://` path on a shared filesystem.

## What a map directory holds

A map is any directory containing a Lanelet2 `.osm`. This is the layout
Autoware publishes, and the one the editor's Map library looks for:

```console
autoware_maps/Town10HD_Opt
├── lanelet2_map.osm          # the map
├── map_projector_info.yaml   # the projection it is read with
└── pointcloud_map.pcd        # not read by this framework, and not downloaded
```

Only the files this framework reads are pulled out of git-lfs — the `.osm`, the
`.xodr` if there is one, and the YAML beside them. The point cloud stays a
pointer file of a few hundred bytes, which is what keeps fetching a 2 MB map
from costing 22 MB.

A repository can carry symlinks, so each file is checked for where it really
is. The `.osm`, the `.xodr` and `map_projector_info.yaml` are used only when
they resolve to a regular file inside the map directory: a link within the
directory is followed, but one that leads out of it, dangles, or names a
directory is ignored, as if the file were absent. An OpenDRIVE read back from
CARLA is held to the same rule inside its derived directory. A `#path` that is
itself a link out of the checkout, or that contains `..`, is refused. Local
directories (`map.directory`) go through the same file checks.

### How a map is read

`map_projector_info.yaml` names the projection the map's lat/lon is read
with, and every loader in this framework -- the run's `MapManager`, the
sweeper, the editor's preview, the ODD route check -- reads it the same way,
and the way Autoware does:

| `projector_type` | Read with |
| --- | --- |
| `MGRS` | `mgrs` (`MGRSProjector`) |
| `LocalCartesianUTM` | `utm` (`UtmProjector` at `map_origin`) |
| `LocalCartesian` | `local_cartesian` (`LocalCartesianProjector` at `map_origin`) |
| `TransverseMercator` | `transverse_mercator` (`TransverseMercatorProjector` at `map_origin`) |
| `Local` | `local_xy` (each node's local tags; see below) |

`LocalCartesianUTM` is read through UTM, not a local tangent plane, despite
its name: the two place a city-sized map tens of metres apart.

The projector is anchored at the whole `map_origin` -- latitude, longitude
and `altitude` -- as Autoware's `get_lanelet2_projector()` anchors it. Only
`LocalCartesian` uses the altitude: it measures every z from it, so near an
origin at 40 m a node's z, and the map-frame poses sent to Autoware, are its
`ele` less 40 m. The other projections ignore it.

A `projector_type: Local` map stores its map frame in each node's `local_x` and
`local_y` tags (and `local_z`, when it has one), which is what Autoware's own
loader reads. So does this framework: such a map is read with the `local_xy`
projection, which takes each node from its local tags and falls back to its
lat/lon (through UTM at the map origin) for a node without them. Set
`map.projector_type: utm` to read every node from its lat/lon instead.

The two agree only where the lat/lon were themselves derived from the local
tags with UTM, as on the official CARLA towns. A map converted by another tool
can differ by about 0.1 % -- 9 cm over 89 m, growing with distance from the
origin -- so the log says which reading was used, and a disagreement beyond
1 cm is logged as a warning.

## Where maps are cached

`~/autoware_data/maps`, overridable with `AUTOWARE_DATA_DIR` (whose `maps`
subdirectory is used) or `AUTOWARE_CARLA_SCENARIO_MAP_CACHE` (used as-is).

That default is the same directory Autoware's own `demo_artifacts` ansible role
downloads map datasets into, unpacked at their repository-relative paths. So on
a machine that has run the Autoware setup, the map is **already there**:

```console
~/autoware_data/maps/autoware_maps/Town10HD_Opt/lanelet2_map.osm
```

and a source naming that path resolves to it in about a millisecond, cloning
nothing. A map this framework did fetch itself lands under `.repos/` in the same
root, so the two never collide and the copy Autoware provisioned always wins.

Nothing re-reads a map's *contents* over the network once it is cached;
`refresh` is the only way to make it fetch again.

One entry under `.repos/` holds one repository at one commit, and every map
in that repository shares it. Its sparse checkout only grows: resolving
`#autoware_maps/TownB` after `#autoware_maps/TownA` adds TownB beside TownA
rather than replacing it, so a map resolved earlier -- held by the editor, a
running case or a sweep over both -- keeps its files. The repository root (a
source with no `#path`) checks the whole tree out. The entry's `meta.json`
lists every directory checked out in it under `materialised`.

One question is asked, though, and it is asked before anything is fetched: when
the ref is a branch, a single `git ls-remote` resolves it to the commit it
points at right now, and everything after that runs as if the source had been
pinned to that commit. So a branch tracks, which is what it reads like -- and,
just as importantly, the cache entry is named after a commit, so a map that
moves lands at a *new path* rather than replacing the bytes under an old one.
Anything holding a parsed map keyed on its path therefore notices.

A machine that cannot reach the remote falls back to the commit that ref last
resolved to, so an offline run still works exactly as it did before.

## Pinning, and why it matters

A source naming a branch means "whatever is on that branch today" -- and, since
the branch is re-resolved on every fetch, it means it literally. Two runs of the
same scenario a month apart are then two runs against two different maps, and
the result of one says nothing about the other.

A source naming a **full commit hash** is pinned: the same URI names the same
bytes forever.

```python
from autoware_carla_scenario.maps import pin_source

pin_source("https://huggingface.co/datasets/OWNER/NAME/tree/main/maps/T")
# 'git+https://huggingface.co/datasets/OWNER/NAME@<40-hex commit>#maps/T'
```

In the editor this is the **Pin to commit** button in the Scenario inspector,
which also tells you when a source is still unpinned.

One consequence is worth stating plainly: an already-unpacked map directory
records no revision, so a **pinned source never uses one**. It always resolves
through git, where the commit is checked rather than assumed. Pinning would be
worthless otherwise.

## OpenDRIVE comes from CARLA

An Autoware map published for a CARLA town ships no `.xodr`, and should not: the
roads belong to the CARLA asset. So the OpenDRIVE is read back out of CARLA,
in the order that costs least:

1. one the repository ships, or one already cached;
2. the asset inside a local CARLA installation, named by the same
   `<MAP_NAME>_PATH` environment variable the rest of the framework uses
   (`TOWN10_HD_OPT_PATH`, and see `.env.example`);
3. a running server, over the CARLA Python API.

Whatever it comes from is cached under `.derived/` beside the map, so it is
fetched once.

**You do not need it to write a scenario.** A scenario is written against
Lanelet2 — a spawn, a goal and a constraint search all name lanelets — so the
editor's preview and a constraint sweep work with no OpenDRIVE and no CARLA
anywhere. A *run* needs it, and a run has CARLA: `ScenarioQueue` writes the
loaded world's OpenDRIVE into the cache the first time a scenario runs on it.

To fetch it ahead of time, use **Fetch OpenDRIVE** in the Scenario inspector, or:

```python
from autoware_carla_scenario.maps import ensure_xodr, resolve_map

ensure_xodr(resolve_map(uri), host="localhost", port=2000)
```

## In the library

### Selecting a published map

```bash
uv run scenario scenario=cut_in/default map=town10hd_opt
```

`conf/map/town10hd_opt.yaml` is a normal map group whose `map.source` names the
published map at a pinned commit. Override it like any other key:

```bash
uv run scenario map=town10hd_opt \
  map.source='git+https://huggingface.co/datasets/OWNER/NAME@REF#maps/T'
```

`map.xodr_path` and `map.lanelet2_path` still win where they are set, so a
single file can be overridden without abandoning the source.

### From Python

```python
from autoware_carla_scenario.maps import list_maps, resolve_map

for entry in list_maps("https://huggingface.co/datasets/AutowareFoundation/carla-ue5-maps@main"):
    print(entry.name, entry.uri)

hd_map = resolve_map(
    "git+https://huggingface.co/datasets/AutowareFoundation/carla-ue5-maps"
    "@main#autoware_maps/Town10HD_Opt"
)
hd_map.lanelet2_path   # -> Path to the .osm, downloaded if it was not there
hd_map.commit          # -> the revision it is at
```

`autoware_carla_scenario.maps` imports neither CARLA nor Lanelet2, so it is safe
to use from any process.

### From a local directory

A map that is on this machine but in no repository -- a copy handed over on a
USB stick, a map still being edited -- is named by its directory:

```bash
uv run scenario map=town10hd_opt map.directory=/path/to/autoware_maps/Town10HD_Opt
```

The directory is read exactly as a map directory in a repository is: its `.osm`
(`lanelet2_map.osm` when there are several), `map_projector_info.yaml`, and any
`.xodr` it ships. It may name the `.osm` itself to pick one out of several.
Nothing is cloned and no revision is recorded, so such a map cannot be pinned.

`map.directory` wins over `map.source`, which is what lets a run be pointed at a
local copy without first clearing the source its map group selects.
`map.xodr_path` and `map.lanelet2_path` still win over both.

An OpenDRIVE taken from CARLA for it is cached under
`.derived/local/<name>-<hash>` in the map root rather than written into the
directory, which may well be read-only.

```python
from autoware_carla_scenario.maps import resolve_directory

hd_map = resolve_directory("~/maps/Town10HD_Opt")
hd_map.lanelet2_path   # -> ~/maps/Town10HD_Opt/lanelet2_map.osm
hd_map.local           # -> True
```

## In the editor

**Map library** in the header browses a repository and lists every map in it.
Each one either starts a new scenario or, when you arrived from a draft, becomes
that scenario's map.

The Scenario inspector carries the same thing per scenario: the source, whether
it is downloaded and whether it is pinned, and buttons to download, refresh, pin
and fetch OpenDRIVE.

Its **Map directory** field is `map.directory`. Setting one clears the Lanelet2
and OpenDRIVE file fields unless they were edited in the same change, and takes
the CARLA map name from the directory. The editor only serves a map from inside
`SCENARIO_EDITOR_MAP_ROOTS` (the working directory unless set), so a directory
elsewhere has to be added there to be previewed.

## Troubleshooting

| Symptom | Cause |
| --- | --- |
| `git is not installed` | The cache clones with `git`; install it. |
| A map resolves but the `.osm` is a few hundred bytes | git-lfs did not pull it. Install `git-lfs`, then **Refresh**. |
| `does not name a directory in the repository` | The `#path` is wrong for that ref. Browse the repository in the Map library. |
| `names a directory outside the repository` / `names a link to a directory outside the repository` | The `#path` contains `..`, or is a symlink leading out of the checkout. Point it at the real directory. |
| A map's `.xodr` or `map_projector_info.yaml` is ignored | It is a symlink to a file outside the map directory, a dangling link, or a link to a directory. |
| A pinned source re-clones on a machine that has the map | Expected: a pinned source never trusts an unpacked directory, because it records no revision. |
