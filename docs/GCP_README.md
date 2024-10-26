
# Project Setup Guide

This guide provides instructions for setting up a Google Cloud VM, generating a clean `requirements.txt`, creating SSH keys, cloning a Git repository via SSH, and running the project with `nohup` while checking logs.

## 1. Creating a VM Instance on Google Cloud
Follow these steps to create a VM instance using the configuration shown in the screenshot:

1. Go to the [Google Cloud Console](https://console.cloud.google.com/).
2. Navigate to **Compute Engine** > **VM Instances**.
3. Click on **Create Instance**.
4. Set the following configuration:
   - **Name**: Choose a suitable name (e.g., `instance-<yourname>`).
   - **Region**: Select `us-west-2` (Avoid `us-west-1` as it might be busy, and other regions might not support GPUs).
   - **Machine type**: Set to GPUs mode with CPU being `n1-standard-2`.
   - **GPUs**: Select `1 x NVIDIA Tesla P4` (Do not choose T4, as it may not be available in many regions).
   - **Boot disk**: Choose a suitable OS image, like Ubuntu 20.04 LTS.
   - **Firewall**: Allow both HTTP and HTTPS traffic.
5. Click on **Create** to launch the instance.

## 2. Generating `requirements.txt` with Clean Format
To generate a clean `requirements.txt` file from your conda environment:
**Do this in local before cloning**

1. Activate your conda environment:
   ```
   conda activate <your-env-name>
   ```
2. Run the following command to generate the `requirements.txt`:
   ```
   pip freeze | grep -v "@" > requirements.txt
   ```
   - This will exclude any local file paths from the output.

## 3. Creating SSH Key and Cloning Git via SSH
To set up SSH keys and clone a Git repository:


1. **Go to your VMs SSH terminal and generate an SSH key** on your VM:
   ```
   ssh-keygen -t ed25519 -C "your_email@example.com"
   ```
   - Press **Enter** to save the key to the default location (`~/.ssh/id_ed25519`).
2. Copy the public key:
   ```
   cat ~/.ssh/id_ed25519.pub
   ```
3. Go to your GitHub account settings and add the public key under **SSH and GPG keys**.
4. **Clone the Git repository** using SSH:
   ```
   git clone git@github.com:<user_name>/<repository_name>.git
   ```

## 4. Running the Project with `nohup` and Checking Logs
To run your project with `nohup` and check logs:

1. Use `nohup` to run the project in the background:
   ```
   nohup python flan_training.py --data_portion 1 --output_report 'flan_training_report.txt' &
   ```
   - The `&` runs the process in the background, and output is saved to `nohup.out` by default.
2. To check the log file:
   ```
   tail -f nohup.out
   ```
   - This will display the real-time logs. Use **Ctrl + C** to stop viewing the logs.

Feel free to update this guide as needed for your project!
