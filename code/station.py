import paramiko
import json
import os



class RPftp:
    def __init__(self, params):
        self.transport = paramiko.Transport((params["hostname"], 
                                        params["port"]
                                        ))
        self.transport.connect(username=username, password=password)
        self.sftp = paramiko.SFTPClient.from_transport(transport)

if __name__ == "__main__":
    # test
    # Connection details
    hostname = '10.42.0.42'
    port = 22
    username = 'root'
    password = 'escondido'
    remote_dir = '/mnt/ramdisk/'
    local_dir = '/home/aldo/jupyter/test_ssh_download/'

    # Create local directory if it doesn't exist
    os.makedirs(local_dir, exist_ok=True)

    # Connect via SFTP
    transport = paramiko.Transport((hostname, port))
    transport.connect(username=username, password=password)
    sftp = paramiko.SFTPClient.from_transport(transport)

    # List and download all files
    for filename in sftp.listdir(remote_dir):
        remote_file = remote_dir + filename
        local_file = os.path.join(local_dir, filename)
        sftp.get(remote_file, local_file)
        print(f'Downloaded: {filename}')

    sftp.close()
    transport.close()