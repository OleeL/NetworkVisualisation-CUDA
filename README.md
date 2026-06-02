# Network Visualisation CUDA

CUDA/OpenGL force-directed graph visualisation for Windows. The app loads a graph, runs the layout on the GPU, and opens a GLFW/OpenGL window to draw the result.

## Requirements

- Windows x64
- NVIDIA GPU with a CUDA-capable driver
  - Tested on this machine with an NVIDIA GeForce RTX 3070 and NVIDIA GeForce 970
- Visual Studio 2022 with the C++ desktop workload
- CUDA Toolkit 11.6 with Visual Studio integration
- GLFW static library and headers
  - This repository already includes `Dependencies/GLFW/include` and `Dependencies/GLFW/lib-vc2019/glfw3.lib`

> Note: the Visual Studio project is configured for CUDA 11.6 and the Visual Studio 2022 `v143` toolset because those are installed on this machine. CUDA 11.6 is older than the current MSVC compiler, so the project also enables NVIDIA/MSVC compatibility overrides needed to build locally.

## Build

From the repository root:

```powershell
& "C:\Program Files\Microsoft Visual Studio\2022\Community\MSBuild\Current\Bin\MSBuild.exe" .\network-visualisation-gpu.sln -p:Configuration=Release -p:Platform=x64 -maxCpuCount
```

Debug builds are also available:

```powershell
& "C:\Program Files\Microsoft Visual Studio\2022\Community\MSBuild\Current\Bin\MSBuild.exe" .\network-visualisation-gpu.sln -p:Configuration=Debug -p:Platform=x64 -maxCpuCount
```

## Run

Run the executable from the repository root so relative `Data` paths resolve correctly.

```powershell
.\x64\Release\network-visualisation-gpu.exe Data\SmallGraph.txt 100
```

Arguments for loading an existing graph:

```text
network-visualisation-gpu.exe [fileName] [iterations]
```

Example datasets included in `Data/`:

- `Data/SmallGraph.txt`
- `Data/100Nodes.txt`
- `Data/1000Nodes.txt`
- `Data/5000Nodes.txt`

The graph file format is plain text:

```text
<number_of_nodes> <number_of_edges>
<source_node> <target_node>
<source_node> <target_node>
...
```

## GPU check

This reports the GPU, driver version, and CUDA driver capability. The CUDA Toolkit compiler version can be checked with:

```powershell
nvcc --version
```
