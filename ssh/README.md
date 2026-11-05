# DLC SSH 登录

`activate_connection.sh` 从同事的脚本复制并独立修改，未修改同事原文件。
默认只读取同目录的 `id_rsa.pub`，不会读取或上传私钥。

## 在 DLC 中启动

镜像需要安装 OpenSSH 服务端（有 `sshd`、`ssh-keygen`），以 root 运行，
并挂载此共享目录。DSW 必须能访问 DLC 容器的 2222 端口且允许 SSH 转发。

在 DLC 的任务启动命令中填写这一行：

```bash
bash /cpfs01/thscc/sharestorage/iwc/HithinkOmni/user_workspace/leizhengxing/leizhengxing/swe/verl/ssh/run_dlc_ssh.sh
```

入口会安装公钥并以前台模式运行 SSH 服务。服务日志直接进入 DLC 任务日志；
看到 `Server listening ... port 2222` 后可尝试连接。
无需启动 Conda、vLLM 或额外的 sleep 进程。
任务结束或被平台回收后，SSH 连接也会结束；此脚本不会改变平台的任务回收策略。

任务日志会打印 `DLC container IP(s)`。如果有多个地址，选择 DSW 可达的内网地址；
重建任务后重新确认 IP。

## 在本地电脑连接

先确认本地执行 `ssh AtlantaDSW` 能正常登录跳板机。
把 `ssh_config.example` 的配置加入本地 `/Users/lzx/.ssh/config`，
将 `DLC_IP_FROM_LOG` 替换为本次 DLC 任务的实际内网 IP，然后执行：

```bash
ssh AtlantaDLC-LZX
```

`IdentityFile /Users/lzx/.ssh/id_rsa` 指本地私钥。
DLC 只需要配对的 `id_rsa.pub`；共享目录里的 `id_rsa` 副本不参与启动或连接，
确认本地仍有原私钥后，可以自行移除该共享副本。
`.gitignore` 已排除本目录的 `id_*` 等密钥文件，避免被普通 `git add` 加入版本库。

## 检查与管理

在此目录下运行：

```bash
# 只检查公钥和 sshd 是否存在，不启动服务或修改系统配置。
bash activate_connection.sh check

# 仅用于已经有其他前台任务维持容器存活的场景。
bash activate_connection.sh start
bash activate_connection.sh status
bash activate_connection.sh stop
```

`stop` 会结束本脚本管理的 SSH 服务；若它是 DLC 前台主进程，任务也会随之结束。
后台 `start` 模式的日志在 `/tmp/lzx-dlc-sshd-0-2222.log`。

需要其他端口时，在 DLC 入口命令前添加 `DLC_SSH_PORT=2223`，
并相应修改本地配置的 `Port`。需要其他公钥时，可设置 `DLC_SSH_PUBLIC_KEY=/路径/公钥.pub`。

若连接失败，本地执行 `ssh -vvv AtlantaDLC-LZX`：

- 超时或 `Connection refused`：检查任务仍在运行、IP/端口以及 DSW 到 DLC 的网络。
- `administratively prohibited`：检查 DSW 的 TCP 转发权限。
- `Permission denied (publickey)`：检查本地私钥是否与 DLC 使用的公钥配对，并查看 DLC SSH 日志。
