# PostgreSQL Setup Guide for Windows

This guide will walk you through installing and configuring PostgreSQL on Windows for the project.

## 1. Install PostgreSQL

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

## 2. Start PostgreSQL Server

By default, the PostgreSQL server should start automatically as a Windows Service after installation.

To verify or start it manually:
1. Open the Windows **Start Menu**, search for **Services**, and hit Enter.
2. Scroll down and look for a service named `postgresql-x64-<version>` (e.g., `postgresql-x64-16`).
3. Ensure its **Status** is "Running". If it is not, right-click on it and select **Start**.

## 3. Create the Database

You can create the database using the command line (`psql`) or the graphical interface (`pgAdmin 4`).

### Option A: Using pgAdmin 4 (GUI - Recommended for beginners)
1. Open the Windows **Start Menu**, search for **pgAdmin 4**, and launch it.
2. It will open in your web browser or as a standalone app. Enter the master password you set during installation.
3. In the left sidebar, expand **Servers** -> **PostgreSQL <version>**.
4. Right-click on **Databases** -> **Create** -> **Database...**
5. In the **Database** field, enter: `deepfake_db`
6. Click **Save**.

### Option B: Using SQL Shell (psql)
1. Open the Windows **Start Menu**, search for **SQL Shell (psql)**, and launch it.
2. Press Enter to accept the defaults for Server, Database, Port, and Username.
3. When prompted, enter the password you set for the `postgres` user during installation.
4. Run the following command to create the database:
   ```sql
   CREATE DATABASE deepfake_db;
   ```
5. You can type `\q` and press Enter to exit psql.

## 4. Configure Backend Environment

Now that your database is set up, you need to configure your backend to connect to it.

1. Open your backend `.env` file (located in `Implementation/backend/.env`).
2. Add or update your database connection variables. Typically, it looks something like this (adjust variable names according to your backend configuration):

   ```env
   DB_HOST=localhost
   DB_PORT=5432
   DB_USER=postgres
   DB_PASSWORD=your_password_here
   DB_NAME=deepfake_db
   ```
   *(Replace `your_password_here` with the password you set during installation)*.
