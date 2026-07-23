# Redis Setup Guide (macOS & Windows)

This guide provides instructions for installing and running Redis on both macOS and Windows.

---

## Setting up Redis on macOS

The easiest and recommended way to install Redis on macOS is using **Homebrew**.

### 1. Install Homebrew (if not already installed)
If you don't have Homebrew installed, open your Terminal and run:
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

### 2. Install Redis
Once Homebrew is installed, run the following command in your Terminal:
```bash
brew install redis
```

### 3. Start the Redis Server
You can start Redis in the foreground or as a background service.

**Option A: Start as a background service (Recommended)**
This will start Redis and automatically restart it when you log in.
```bash
brew services start redis
```

**Option B: Start in the foreground**
If you just want to run it for the current session (stopping when you close the terminal):
```bash
redis-server
```

### 4. Test the Redis Connection
Open a new Terminal window and type:
```bash
redis-cli
```
Then, type `ping` and press Enter. You should receive `PONG` as a response, indicating that Redis is running properly.

---

## Setting up Redis on Windows

Redis is not officially supported natively on Windows. However, you have two great options for running it: using **WSL (Windows Subsystem for Linux)** or **Docker**.

### Option 1: Using WSL (Recommended)

WSL allows you to run a Linux environment directly on Windows, which is perfect for running Redis.

#### 1. Install WSL (if not already installed)
Open PowerShell or Command Prompt as **Administrator** and run:
```cmd
wsl --install
```
*Note: This will install Ubuntu by default. You may need to restart your computer after this step.*

#### 2. Install Redis inside WSL
Open your WSL terminal (search for "Ubuntu" in your Start Menu) and run the following commands to update your package list and install Redis:
```bash
sudo apt update
sudo apt install redis-server
```

#### 3. Start the Redis Server
To start the Redis service in your WSL terminal, run:
```bash
sudo service redis-server start
```

#### 4. Test the Redis Connection
In the same WSL terminal, type:
```bash
redis-cli
```
Then, type `ping` and press Enter. You should receive `PONG` as a response.

*(Note: Applications running on your Windows machine can connect to this Redis instance at `localhost:6379`)*.

---

### Option 2: Using Docker

If you already use Docker, this is a very clean way to run Redis without installing anything else directly on your system.

#### 1. Install Docker Desktop
Download and install [Docker Desktop for Windows](https://docs.docker.com/desktop/install/windows-install/).

#### 2. Run a Redis Container
Open PowerShell or Command Prompt and run the following command. This will download the official Redis image and start a container:
```cmd
docker run --name my-redis-container -p 6379:6379 -d redis
```

#### 3. Test the Redis Connection
You can connect to the running container using `redis-cli` by running:
```cmd
docker exec -it my-redis-container redis-cli
```
Type `ping` and press Enter to receive a `PONG`.

---

## 5. Configure Backend Environment

Once Redis is running on either macOS or Windows, ensure your backend `.env` file (located in `Implementation/backend/.env`) has the correct connection URL. The default URL is usually sufficient for local development:

```env
REDIS_URL=redis://localhost:6379/0
```
*(The `/0` at the end specifies the default database index)*.
