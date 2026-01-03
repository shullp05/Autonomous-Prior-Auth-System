# Baseline Toolchain

## uname -a
Linux PSHULL 5.15.167.4-microsoft-standard-WSL2 #1 SMP Tue Nov 5 00:21:55 UTC 2024 x86_64 x86_64 x86_64 GNU/Linux

## python --version
Python 3.11.14

## pip --version
pip 25.3 from /root/micromamba/envs/revenue_agent/lib/python3.11/site-packages/pip (python 3.11)

## pytest --version
pytest 9.0.1

## conda info
/root/micromamba/lib/python3.9/site-packages/conda/base/context.py:211: FutureWarning: Adding 'defaults' to channel list implicitly is deprecated and will be removed in 25.9. 

To remove this warning, please choose a default channel explicitly with conda's regular configuration system, e.g. by adding 'defaults' to the list of channels:

  conda config --add channels defaults

For more information see https://docs.conda.io/projects/conda/en/stable/user-guide/configuration/use-condarc.html

  deprecated.topic(

# >>>>>>>>>>>>>>>>>>>>>> ERROR REPORT <<<<<<<<<<<<<<<<<<<<<<

    Traceback (most recent call last):
      File "/root/micromamba/lib/python3.9/site-packages/conda/core/index.py", line 185, in system_packages
        return self._system_packages
    AttributeError: 'Index' object has no attribute '_system_packages'
    
    During handling of the above exception, another exception occurred:
    
    Traceback (most recent call last):
      File "/root/micromamba/lib/python3.9/site-packages/conda/exception_handler.py", line 28, in __call__
        return func(*args, **kwargs)
      File "/root/micromamba/lib/python3.9/site-packages/conda/cli/main.py", line 61, in main_subshell
        exit_code = do_call(args, parser)
      File "/root/micromamba/lib/python3.9/site-packages/conda/cli/conda_argparse.py", line 206, in do_call
        result = getattr(module, func_name)(args, parser)
      File "/root/micromamba/lib/python3.9/site-packages/conda/cli/main_info.py", line 564, in execute
        renderer = InfoRenderer(context)
      File "/root/micromamba/lib/python3.9/site-packages/conda/cli/main_info.py", line 423, in __init__
        self._info_dict = get_info_dict()
      File "/root/micromamba/lib/python3.9/site-packages/conda/cli/main_info.py", line 226, in get_info_dict
        virtual_pkg_index = Index().system_packages
      File "/root/micromamba/lib/python3.9/site-packages/conda/core/index.py", line 187, in system_packages
        self.reload(system=True)
      File "/root/micromamba/lib/python3.9/site-packages/conda/core/index.py", line 242, in reload
        for package in context.plugin_manager.get_virtual_package_records()
      File "/root/micromamba/lib/python3.9/site-packages/conda/plugins/manager.py", line 453, in get_virtual_package_records
        for hook in self.get_hook_results("virtual_packages")
      File "/root/micromamba/lib/python3.9/site-packages/conda/plugins/manager.py", line 291, in get_hook_results
        plugins = [plugin for plugins in hook(**kwargs) for plugin in plugins]
      File "/root/micromamba/lib/python3.9/site-packages/conda/plugins/manager.py", line 291, in <listcomp>
        plugins = [plugin for plugins in hook(**kwargs) for plugin in plugins]
      File "/root/micromamba/lib/python3.9/site-packages/conda/plugins/virtual_packages/cuda.py", line 66, in conda_virtual_packages
        cuda_version = cached_cuda_version()
      File "/root/micromamba/lib/python3.9/site-packages/conda/plugins/virtual_packages/cuda.py", line 61, in cached_cuda_version
        return cuda_version()
      File "/root/micromamba/lib/python3.9/site-packages/conda/plugins/virtual_packages/cuda.py", line 36, in cuda_version
        queue = context.SimpleQueue()
      File "/root/micromamba/lib/python3.9/multiprocessing/context.py", line 113, in SimpleQueue
        return SimpleQueue(ctx=self.get_context())
      File "/root/micromamba/lib/python3.9/multiprocessing/queues.py", line 341, in __init__
        self._rlock = ctx.Lock()
      File "/root/micromamba/lib/python3.9/multiprocessing/context.py", line 68, in Lock
        return Lock(ctx=self.get_context())
      File "/root/micromamba/lib/python3.9/multiprocessing/synchronize.py", line 162, in __init__
        SemLock.__init__(self, SEMAPHORE, 1, 1, ctx=ctx)
      File "/root/micromamba/lib/python3.9/multiprocessing/synchronize.py", line 57, in __init__
        sl = self._semlock = _multiprocessing.SemLock(
    PermissionError: [Errno 13] Permission denied

`$ /root/micromamba/condabin/conda info`


An unexpected error has occurred. Conda has prepared the above report.
If you suspect this error is being caused by a malfunctioning plugin,
consider using the --no-plugins option to turn off plugins.

Example: conda --no-plugins install <package>

Alternatively, you can set the CONDA_NO_PLUGINS environment variable on
the command line to run the command without plugins enabled.

Example: CONDA_NO_PLUGINS=true conda install <package>

If submitted, this report will be used by core maintainers to improve
future releases of conda.
Would you like conda to send this report to the core maintainers? [y/N]: 
No report sent. To permanently opt-out, use

    $ conda config --set report_errors false


## conda list
# packages in environment at /root/micromamba/envs/revenue_agent:
#
# Name                                      Version          Build            Channel
_libgcc_mutex                               0.1              main
_openmp_mutex                               5.1              1_gnu
aiohappyeyeballs                            2.6.1            pypi_0           pypi
aiohttp                                     3.13.2           pypi_0           pypi
aiosignal                                   1.4.0            pypi_0           pypi
annotated-doc                               0.0.4            pypi_0           pypi
annotated-types                             0.7.0            pypi_0           pypi
anyio                                       4.12.0           pypi_0           pypi
attrs                                       25.4.0           pypi_0           pypi
backoff                                     2.2.1            pypi_0           pypi
bandit                                      1.9.2            pypi_0           pypi
bcembedding                                 0.1.5            pypi_0           pypi
bcrypt                                      5.0.0            pypi_0           pypi
blinker                                     1.9.0            pypi_0           pypi
build                                       1.3.0            pypi_0           pypi
bzip2                                       1.0.8            h5eee18b_6
ca-certificates                             2025.11.4        h06a4308_0
cachetools                                  6.2.2            pypi_0           pypi
certifi                                     2025.11.12       pypi_0           pypi
charset-normalizer                          3.4.4            pypi_0           pypi
chromadb                                    1.3.5            pypi_0           pypi
click                                       8.3.1            pypi_0           pypi
coloredlogs                                 15.0.1           pypi_0           pypi
coverage                                    7.12.0           pypi_0           pypi
dataclasses-json                            0.6.7            pypi_0           pypi
datasets                                    4.4.1            pypi_0           pypi
dill                                        0.4.0            pypi_0           pypi
distro                                      1.9.0            pypi_0           pypi
durationpy                                  0.10             pypi_0           pypi
expat                                       2.7.3            h3385a95_0
fastapi                                     0.124.4          pypi_0           pypi
filelock                                    3.20.0           pypi_0           pypi
flask                                       3.1.2            pypi_0           pypi
flask-cors                                  6.0.2            pypi_0           pypi
flatbuffers                                 25.9.23          pypi_0           pypi
frozenlist                                  1.8.0            pypi_0           pypi
fsspec                                      2025.10.0        pypi_0           pypi
google-auth                                 2.43.0           pypi_0           pypi
googleapis-common-protos                    1.72.0           pypi_0           pypi
greenlet                                    3.2.4            pypi_0           pypi
grpcio                                      1.76.0           pypi_0           pypi
h11                                         0.16.0           pypi_0           pypi
hf-xet                                      1.2.0            pypi_0           pypi
httpcore                                    1.0.9            pypi_0           pypi
httptools                                   0.7.1            pypi_0           pypi
httpx                                       0.28.1           pypi_0           pypi
httpx-sse                                   0.4.3            pypi_0           pypi
huggingface-hub                             0.36.0           pypi_0           pypi
humanfriendly                               10.0             pypi_0           pypi
idna                                        3.11             pypi_0           pypi
importlib-metadata                          8.7.0            pypi_0           pypi
importlib-resources                         6.5.2            pypi_0           pypi
iniconfig                                   2.3.0            pypi_0           pypi
itsdangerous                                2.2.0            pypi_0           pypi
jinja2                                      3.1.6            pypi_0           pypi
joblib                                      1.5.2            pypi_0           pypi
jsonpatch                                   1.33             pypi_0           pypi
jsonpointer                                 3.0.0            pypi_0           pypi
jsonschema                                  4.25.1           pypi_0           pypi
jsonschema-specifications                   2025.9.1         pypi_0           pypi
kubernetes                                  34.1.0           pypi_0           pypi
langchain                                   1.1.0            pypi_0           pypi
langchain-chroma                            1.0.0            pypi_0           pypi
langchain-classic                           1.0.0            pypi_0           pypi
langchain-community                         0.4.1            pypi_0           pypi
langchain-core                              1.1.0            pypi_0           pypi
langchain-ollama                            1.0.0            pypi_0           pypi
langchain-text-splitters                    1.0.0            pypi_0           pypi
langgraph                                   1.0.4            pypi_0           pypi
langgraph-checkpoint                        3.0.1            pypi_0           pypi
langgraph-prebuilt                          1.0.5            pypi_0           pypi
langgraph-sdk                               0.2.10           pypi_0           pypi
langsmith                                   0.4.49           pypi_0           pypi
ld_impl_linux-64                            2.44             h153f514_2
libffi                                      3.4.4            h6a678d5_1
libgcc                                      15.2.0           h69a1729_7
libgcc-ng                                   15.2.0           h166f726_7
libgomp                                     15.2.0           h4751f2c_7
libnsl                                      2.0.0            h5eee18b_0
libstdcxx                                   15.2.0           h39759b7_7
libstdcxx-ng                                15.2.0           hc03a8fd_7
libuuid                                     1.41.5           h5eee18b_0
libxcb                                      1.17.0           h9b100fa_0
libzlib                                     1.3.1            hb25bd0a_0
markdown-it-py                              4.0.0            pypi_0           pypi
markupsafe                                  3.0.3            pypi_0           pypi
marshmallow                                 3.26.1           pypi_0           pypi
mdurl                                       0.1.2            pypi_0           pypi
mmh3                                        5.2.0            pypi_0           pypi
mpmath                                      1.3.0            pypi_0           pypi
multidict                                   6.7.0            pypi_0           pypi
multiprocess                                0.70.18          pypi_0           pypi
mypy-extensions                             1.1.0            pypi_0           pypi
ncurses                                     6.5              h7934f7d_0
networkx                                    3.6.1            pypi_0           pypi
numpy                                       1.26.4           pypi_0           pypi
nvidia-cublas-cu12                          12.8.4.1         pypi_0           pypi
nvidia-cuda-cupti-cu12                      12.8.90          pypi_0           pypi
nvidia-cuda-nvrtc-cu12                      12.8.93          pypi_0           pypi
nvidia-cuda-runtime-cu12                    12.8.90          pypi_0           pypi
nvidia-cudnn-cu12                           9.10.2.21        pypi_0           pypi
nvidia-cufft-cu12                           11.3.3.83        pypi_0           pypi
nvidia-cufile-cu12                          1.13.1.3         pypi_0           pypi
nvidia-curand-cu12                          10.3.9.90        pypi_0           pypi
nvidia-cusolver-cu12                        11.7.3.90        pypi_0           pypi
nvidia-cusparse-cu12                        12.5.8.93        pypi_0           pypi
nvidia-cusparselt-cu12                      0.7.1            pypi_0           pypi
nvidia-nccl-cu12                            2.27.5           pypi_0           pypi
nvidia-nvjitlink-cu12                       12.8.93          pypi_0           pypi
nvidia-nvshmem-cu12                         3.3.20           pypi_0           pypi
nvidia-nvtx-cu12                            12.8.90          pypi_0           pypi
oauthlib                                    3.3.1            pypi_0           pypi
ollama                                      0.6.1            pypi_0           pypi
onnxruntime                                 1.23.2           pypi_0           pypi
openssl                                     3.0.18           hd6dcaed_0
opentelemetry-api                           1.38.0           pypi_0           pypi
opentelemetry-exporter-otlp-proto-common    1.38.0           pypi_0           pypi
opentelemetry-exporter-otlp-proto-grpc      1.38.0           pypi_0           pypi
opentelemetry-proto                         1.38.0           pypi_0           pypi
opentelemetry-sdk                           1.38.0           pypi_0           pypi
opentelemetry-semantic-conventions          0.59b0           pypi_0           pypi
orjson                                      3.11.4           pypi_0           pypi
ormsgpack                                   1.12.0           pypi_0           pypi
overrides                                   7.7.0            pypi_0           pypi
packaging                                   25.0             pypi_0           pypi
pandas                                      2.3.3            pypi_0           pypi
pillow                                      12.0.0           pypi_0           pypi
pip                                         25.3             pyhc872135_0
pluggy                                      1.6.0            pypi_0           pypi
posthog                                     5.4.0            pypi_0           pypi
propcache                                   0.4.1            pypi_0           pypi
protobuf                                    6.33.1           pypi_0           pypi
psutil                                      7.1.3            pypi_0           pypi
pthread-stubs                               0.3              h0ce48e5_1
pyarrow                                     22.0.0           pypi_0           pypi
pyasn1                                      0.6.1            pypi_0           pypi
pyasn1-modules                              0.4.2            pypi_0           pypi
pybase64                                    1.4.2            pypi_0           pypi
pydantic                                    2.12.5           pypi_0           pypi
pydantic-core                               2.41.5           pypi_0           pypi
pydantic-settings                           2.12.0           pypi_0           pypi
pygments                                    2.19.2           pypi_0           pypi
pypdf                                       6.4.0            pypi_0           pypi
pypika                                      0.48.9           pypi_0           pypi
pyproject-hooks                             1.2.0            pypi_0           pypi
pytest                                      9.0.1            pypi_0           pypi
pytest-cov                                  7.0.0            pypi_0           pypi
python                                      3.11.14          h6fa692b_0
python-dateutil                             2.9.0.post0      pypi_0           pypi
python-dotenv                               1.2.1            pypi_0           pypi
pytz                                        2025.2           pypi_0           pypi
pyyaml                                      6.0.3            pypi_0           pypi
readline                                    8.3              hc2a1206_0
referencing                                 0.37.0           pypi_0           pypi
regex                                       2025.11.3        pypi_0           pypi
reportlab                                   4.4.5            pypi_0           pypi
requests                                    2.32.5           pypi_0           pypi
requests-oauthlib                           2.0.0            pypi_0           pypi
requests-toolbelt                           1.0.0            pypi_0           pypi
rich                                        14.2.0           pypi_0           pypi
rpds-py                                     0.29.0           pypi_0           pypi
rsa                                         4.9.1            pypi_0           pypi
ruff                                        0.14.9           pypi_0           pypi
safetensors                                 0.7.0            pypi_0           pypi
scikit-learn                                1.8.0            pypi_0           pypi
scipy                                       1.16.3           pypi_0           pypi
sentence-transformers                       3.0.1            pypi_0           pypi
setuptools                                  80.9.0           py311h06a4308_0
shellingham                                 1.5.4            pypi_0           pypi
six                                         1.17.0           pypi_0           pypi
sqlalchemy                                  2.0.44           pypi_0           pypi
sqlite                                      3.51.0           h2a70700_0
starlette                                   0.50.0           pypi_0           pypi
stevedore                                   5.6.0            pypi_0           pypi
sympy                                       1.14.0           pypi_0           pypi
tenacity                                    9.1.2            pypi_0           pypi
threadpoolctl                               3.6.0            pypi_0           pypi
tk                                          8.6.15           h54e0aa7_0
tokenizers                                  0.15.2           pypi_0           pypi
torch                                       2.9.1            pypi_0           pypi
tqdm                                        4.67.1           pypi_0           pypi
transformers                                4.36.2           pypi_0           pypi
triton                                      3.5.1            pypi_0           pypi
typer                                       0.20.0           pypi_0           pypi
typer-slim                                  0.20.0           pypi_0           pypi
typing-extensions                           4.15.0           pypi_0           pypi
typing-inspect                              0.9.0            pypi_0           pypi
typing-inspection                           0.4.2            pypi_0           pypi
tzdata                                      2025.2           pypi_0           pypi
urllib3                                     2.3.0            pypi_0           pypi
uvicorn                                     0.38.0           pypi_0           pypi
uvloop                                      0.22.1           pypi_0           pypi
watchfiles                                  1.1.1            pypi_0           pypi
websocket-client                            1.9.0            pypi_0           pypi
websockets                                  15.0.1           pypi_0           pypi
werkzeug                                    3.1.4            pypi_0           pypi
wheel                                       0.45.1           py311h06a4308_0
xorg-libx11                                 1.8.12           h9b100fa_1
xorg-libxau                                 1.0.12           h9b100fa_0
xorg-libxdmcp                               1.1.5            h9b100fa_0
xorg-xorgproto                              2024.1           h5eee18b_1
xxhash                                      3.6.0            pypi_0           pypi
xz                                          5.6.4            h5eee18b_1
yarl                                        1.22.0           pypi_0           pypi
zipp                                        3.23.0           pypi_0           pypi
zlib                                        1.3.1            hb25bd0a_0
zstandard                                   0.25.0           pypi_0           pypi

## docker --version

The command 'docker' could not be found in this WSL 2 distro.
We recommend to activate the WSL integration in Docker Desktop settings.

For details about using Docker Desktop with WSL 2, visit:

https://docs.docker.com/go/wsl2/


## node --version
v22.19.0

## npm --version
10.9.3
