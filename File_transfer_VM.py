import os
import sys
import shutil
import paramiko
from google.cloud import storage

os.system('clear')
# ask for the external VM IP
#vm_ip = input("External IP of you VM: ")
vm_ip = '34.91.51.192'

# vm credentials
vm_username = 'kirafriedrichs'
private_key_path = '/Users/friedrichs_kira/.ssh/google_compute_engine'
home_path = '/home/kirafriedrichs/'
local_path = './'

# open ssh
# Create a SSH client
client = paramiko.SSHClient()
# Automatically add the server's host key
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
# Load the private key
key = paramiko.RSAKey.from_private_key_file(private_key_path)
# Connect to the VM instance
client.connect(hostname=vm_ip, username=vm_username, pkey=key)

# directories
dir_vm = ['./']

# ask for the directory on the vm
print("")
print("You can choose from the directories: ")
print("")
j = 0
for dir in dir_vm:
    j += 1
    print(f"[{j}] {dir}")
    
print("")
dir_key = input("Please choose number of directory: ")
dir_path = home_path + dir_vm[int(dir_key) - 1]
print
print("Selected directory: ", dir_path)

# List files
# Execute the ls command on the remote machine and get the output
stdin, stdout, stderr = client.exec_command(f'ls {dir_path}')
output = stdout.readlines()
#type(output)
file_list_vm = []
for files_vm in output:
    file_list_vm.append(files_vm.replace('\n',''))
    

all_files_vm = []
for i in file_list_vm:
    #print(i)
    if "." in i or "last_checkpoint" in i:
        all_files_vm.append(i.split("."))
        
all_files_vm = sorted(all_files_vm, key = lambda x: x[-1])

print("")
print(f"The folder {dir_path} has the entries: \n")

file_list_sorted = []
k = 0
for files in all_files_vm:
    k += 1
    print(f"[{k}] {'.'.join(files)}")
    file_list_sorted.append('.'.join(files))
    
#print(file_list_sorted)

# Select the files you want to copy
print("")
select_file_keys = input("Which files do you want to copy? ")
print("")
print("You selected the following file(s): \n")
#print(select_file_keys)
all_files_vm_final = []
for l in select_file_keys.split(","):
    #print(l)
    print(file_list_sorted[int(l)-1])
    all_files_vm_final.append(file_list_sorted[int(l)-1])
    
    # Copy the files to your local machine
print("")
cont = input("Do you want to continue? [y/N] ")

if cont == "y":
    #print("Weitermachen")
    ftp_client = client.open_sftp()
    for f_vm in all_files_vm_final:
        #print(f_vm)
        #print("")
        print(f"Copy {dir_path}" + f"{f_vm} from VM to local {local_path}")
    # Copy the file from the VM instance to the local machine
        #print(f"Copy {f_vm} to {local_path}")
        ftp_client.get(dir_path + f_vm, local_path + f_vm)
    ftp_client.close()
else:
    print("Stopping the script")
#     print("Stoppen")
#     #sys.exit("Either you said no or you didn't hit the y")
#     print("danach")
    
#     # Close the SSH client
# ##client.close()
