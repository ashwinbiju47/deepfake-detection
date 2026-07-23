# PostgreSQL Setup Guide (macOS & Windows)

This guide provides instructions for installing and configuring PostgreSQL on both macOS and Windows for this project.

---

## Setting up PostgreSQL on macOS

There are two primary ways to install PostgreSQL on macOS: using the graphical **Postgres.app** (recommended for ease of use) or using **Homebrew**.

### Option 1: Using Postgres.app (Recommended)
1. Go to [https://postgresapp.com/](https://postgresapp.com/) and download the latest release.
2. Open the downloaded `.dmg` file and drag the **Postgres** icon to your **Applications** folder.
3. Open Postgres from your Applications folder. You may get a warning about an app downloaded from the internet; click **Open**.
4. Click **Initialize** to create a new database server.
5. Ensure the server is running (the elephant icon in your menu bar will show as active).
6. Double-click on any of the default databases listed in the app to open the `psql` command-line interface, or you can run `psql` in your terminal if you configure your PATH.

### Option 2: Using Homebrew
1. If you don't have Homebrew installed, install it first via your terminal:
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```
2. Install PostgreSQL:
   ```bash
   brew install postgresql
   ```
3. Start the PostgreSQL service:
   ```bash
   brew services start postgresql
   ```

---

## Setting up PostgreSQL on Windows

1. **Download the Installer:**
   - Go to the official PostgreSQL download page for Windows: [https://www.postgresql.org/download/windows/](https://www.postgresql.org/download/windows/)
   - Click on "Download the installer" and download the latest version for Windows x86-64.

2. **Run the Installer:**
   - Double-click the downloaded `.exe` file.
   - Follow the setup wizard.
   - **Select Components:** Leave all default components checked (PostgreSQL Server, pgAdmin 4, Stack Builder, Command Line Tools).
   - **Data Directory:** You can leave the default data directory.
   - **Password:** You will be prompted to provide a password for the database superuser (`postgres`). **Note this password down**, as you will need it later.
   - **Port:** Leave the default port at `5432`.
   - **Advanced Options:** Leave the default locale.
   - Complete the installation. You can skip Stack Builder at the end by unchecking the box before clicking Finish.

3. **Start PostgreSQL Server:**
   - By default, the PostgreSQL server should start automatically as a Windows Service after installation.
   - To verify, open the Windows **Start Menu**, search for **Services**, find `postgresql-x64-<version>`, and ensure its **Status** is "Running".

---

## Creating the Project Database

Once PostgreSQL is installed and running on either OS, you need to create the database for the project (`deepfake_db`).

### Using the Command Line (psql)
1. Open your terminal (macOS) or SQL Shell / psql (Windows).
2. Connect to the default `postgres` database:
   - macOS (Postgres.app): Double-click the `postgres` database in the app UI.
   - macOS (Homebrew): Run `psql postgres` in your terminal.
   - Windows: Launch **SQL Shell (psql)** from the Start Menu, press Enter to accept defaults, and enter your password.
3. Run the following command to create the database:
   ```sql
   CREATE DATABASE deepfake_db;
   ```
4. Type `\q` and press Enter to exit `psql`.

### Using a GUI (pgAdmin 4 - Included with Windows installer)
1. Open **pgAdmin 4**.
2. Connect to your local server (enter your password if prompted).
3. Right-click on **Databases** -> **Create** -> **Database...**
4. Enter `deepfake_db` as the database name and click **Save**.

---

## Configuring Backend Environment

Finally, configure your backend to connect to the new database.

1. Open your backend `.env` file (located in `Implementation/backend/.env`).
2. Add or update your database connection variables according to your OS setup:

**If using macOS (typically no password for the default user):**
```env
DB_HOST=localhost
DB_PORT=5432
DB_USER=your_mac_username # usually your system username, or postgres
DB_PASSWORD=
DB_NAME=deepfake_db
```

**If using Windows (uses the password you set during installation):**
```env
DB_HOST=localhost
DB_PORT=5432
DB_USER=postgres
DB_PASSWORD=your_password_here
DB_NAME=deepfake_db
```
