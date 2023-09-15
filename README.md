# Rev 4 - Chapter 2 Demo

This repository holds the training code that was showcased during the Rev4 keynote. After following the setup instructions below, you can run `Demo.ipynb` to reproduce the same results.

## Data Setup

In this demo, we uploaded our data into a Domino Dataset called `financial-news` and reference the mounted dataset path in the training code.

The data has been included in this repo as `data.csv` for your convenience. You can choose to upload it as a dataset or have the code point directly to it.

## Environment Setup 

This project requires the following [compute environments](https://docs.dominodatalab.com/en/latest/user_guide/f51038/environments/) to be present:

### Ray Workspace 2.2.0

**Environment Base** 

`quay.io/domino/compute-environment-images:ubuntu20-py3.9-r4.2-domino5.6-gpu`

**Dockerfile Instructions**

```
RUN pip install datasets==2.12.0 transformers==4.29.0 torch==2.0.1 ray[all]==2.4.0 accelerate==0.19.0 streamlit==1.22.0 tblib==1.7.0 pandas==2.0.1 https://vve589t3tspu.s3.us-west-2.amazonaws.com/5/domino_code_assist-1.3.0-py2.py3-none-any.whl
```

**Pluggable Workspace Tools**

```
jupyter:
  title: "Jupyter (Python, R, Julia)"
  iconUrl: "/assets/images/workspace-logos/Jupyter.svg"
  start: [ "/opt/domino/workspaces/jupyter/start" ]
  supportedFileExtensions: [ ".ipynb" ]
  httpProxy:
    port: 8888
    rewrite: false
    internalPath: "/{{ownerUsername}}/{{projectName}}/{{sessionPathComponent}}/{{runId}}/{{#if pathToOpen}}tree/{{pathToOpen}}{{/if}}"
    requireSubdomain: false
jupyterlab:
  title: "JupyterLab"
  iconUrl: "/assets/images/workspace-logos/jupyterlab.svg"
  start: [  "/opt/domino/workspaces/jupyterlab/start" ]
  httpProxy:
    internalPath: "/{{ownerUsername}}/{{projectName}}/{{sessionPathComponent}}/{{runId}}/{{#if pathToOpen}}tree/{{pathToOpen}}{{/if}}"
    port: 8888
    rewrite: false
    requireSubdomain: false
vscode:
  title: "vscode"
  iconUrl: "/assets/images/workspace-logos/vscode.svg"
  start: [ "/opt/domino/workspaces/vscode/start" ]
  httpProxy:
    port: 8888
    requireSubdomain: false
rstudio:
  title: "RStudio"
  iconUrl: "/assets/images/workspace-logos/Rstudio.svg"
  start: [ "/opt/domino/workspaces/rstudio/start" ]
  httpProxy:
    port: 8888
    requireSubdomain: false
```

### Ray Cluster 2.2.0

**Environment Base** 

`rayproject/ray-ml:2.4.0-py39`

**Supported Cluster Settings**

Ray

**Dockerfile Instructions**
```
RUN pip install datasets==2.12.0 transformers==4.29.0 torch==2.0.1 accelerate==0.19.0 scikit-learn
USER root
RUN usermod -u 12574 ray 
```