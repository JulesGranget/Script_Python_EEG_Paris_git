

import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import mne
import pandas as pd
import sys
import stat
import subprocess
import scipy.stats
import xarray as xr
import physio
import getpass

from bycycle.cyclepoints import find_extrema
import neurokit2 as nk

from n0_config_params import *


debug = False




#sig = data
def iirfilt(sig, srate, lowcut=None, highcut=None, order=4, ftype='butter', verbose=False, show=False, axis=0):

    if len(sig.shape) == 1:

        axis = 0

    if lowcut is None and not highcut is None:
        btype = 'lowpass'
        cut = highcut

    if not lowcut is None and highcut is None:
        btype = 'highpass'
        cut = lowcut

    if not lowcut is None and not highcut is None:
        btype = 'bandpass'

    if btype in ('bandpass', 'bandstop'):
        band = [lowcut, highcut]
        assert len(band) == 2
        Wn = [e / srate * 2 for e in band]
    else:
        Wn = float(cut) / srate * 2

    filter_mode = 'sos'
    sos = scipy.signal.iirfilter(order, Wn, analog=False, btype=btype, ftype=ftype, output=filter_mode)

    filtered_sig = scipy.signal.sosfiltfilt(sos, sig, axis=axis)

    return filtered_sig


########################################
######## SURFACE LAPLACIAN ########
########################################


#raw, leg_order, m, smoothing = raw, 4, 50, 1e-5
def surface_laplacian(raw, leg_order, m, smoothing):
    """
    This function attempts to compute the surface laplacian transform to an mne Epochs object. The 
    algorithm follows the formulations of Perrin et al. (1989) and it consists for the most part in a 
    nearly-literal translation of Mike X Cohen's 'Analyzing neural time series data' corresponding MATLAB 
    code (2014).
    
    INPUTS are:
        - raw: raw mne object with  data(chan,sig)
        - leg_order: maximum order of the Legendre polynomial
        - m: smothness parameter for G and H
        - smoothing: smothness parameter for the diagonal of G
        - montage: montage to reconstruct the transformed Epochs object (same as in raw data import)
        
    OUTPUTS are:
        - raw_lap: surface laplacian transform of the original raw object
        
    References:
        - Perrin, F., Pernier, J., Bertrand, O. & Echallier, J.F. (1989). Spherical splines for scalp 
          potential and current density mapping. Electroencephalography and clinical Neurophysiology, 72, 
          184-187.
        - Cohen, M.X. (2014). Surface Laplacian In Analyzing neural time series data: theory and practice 
          (pp. 275-290). London, England: The MIT Press.
    """
    # import libraries
    import numpy as np
    from scipy import special
    import math
    import mne
    
    # get electrodes positions
    locs = raw._get_channel_positions()

    x = locs[:,0]
    y = locs[:,1]
    z = locs[:,2]

    # arrange data
    data = raw.get_data() # data
    orig_data_size = np.squeeze(data.shape)

    numelectrodes = len(x)
    
    # normalize cartesian coordenates to sphere unit
    def cart2sph(x, y, z):
        hxy = np.hypot(x, y)
        r = np.hypot(hxy, z)
        el = np.arctan2(z, hxy)
        az = np.arctan2(y, x)
        return az, el, r

    junk1, junk2, spherical_radii = cart2sph(x,y,z)
    maxrad = np.max(spherical_radii)
    x = x/maxrad
    y = y/maxrad
    z = z/maxrad
    
    # compute cousine distance between all pairs of electrodes
    cosdist = np.zeros((numelectrodes, numelectrodes))
    for i in range(numelectrodes):
        for j in range(i+1,numelectrodes):
            cosdist[i,j] = 1 - (((x[i] - x[j])**2 + (y[i] - y[j])**2 + (z[i] - z[j])**2)/2)

    cosdist = cosdist + cosdist.T + np.identity(numelectrodes)

    # get legendre polynomials
    legpoly = np.zeros((leg_order, numelectrodes, numelectrodes))
    for ni in range(leg_order):
        for i in range(numelectrodes):
            for j in range(i+1, numelectrodes):
                #temp = special.lpn(8,cosdist[0,1])[0][8]
                legpoly[ni,i,j] = special.lpn(ni+1,cosdist[i,j])[0][ni+1]

    legpoly = legpoly + np.transpose(legpoly,(0,2,1))

    for i in range(leg_order):
        legpoly[i,:,:] = legpoly[i,:,:] + np.identity(numelectrodes)

    # compute G and H matrixes
    twoN1 = np.multiply(2, range(1, leg_order+1))+1
    gdenom = np.power(np.multiply(range(1, leg_order+1), range(2, leg_order+2)), m, dtype=float)
    hdenom = np.power(np.multiply(range(1, leg_order+1), range(2, leg_order+2)), m-1, dtype=float)

    G = np.zeros((numelectrodes, numelectrodes))
    H = np.zeros((numelectrodes, numelectrodes))

    for i in range(numelectrodes):
        for j in range(i, numelectrodes):

            g = 0
            h = 0

            for ni in range(leg_order):
                g = g + (twoN1[ni] * legpoly[ni,i,j]) / gdenom[ni]
                h = h - (twoN1[ni] * legpoly[ni,i,j]) / hdenom[ni]

            G[i,j] = g / (4*np.pi)
            H[i,j] = -h / (4*np.pi)

    G = G + G.T
    H = H + H.T

    G = G - np.identity(numelectrodes) * G[1,1] / 2
    H = H - np.identity(numelectrodes) * H[1,1] / 2

    # compute C matrix
    Gs = G + np.identity(numelectrodes) * smoothing
    GsinvS = np.sum(np.linalg.inv(Gs), 0)
    dataGs = np.dot(data.T, np.linalg.inv(Gs))
    C = dataGs - np.dot(np.atleast_2d(np.sum(dataGs, 1)/np.sum(GsinvS)).T, np.atleast_2d(GsinvS))

    # apply transform
    original = np.reshape(data, orig_data_size)
    surf_lap = np.reshape(np.transpose(np.dot(C,np.transpose(H))), orig_data_size)

    info = raw.info
    raw_lap =  mne.io.RawArray(surf_lap,info)
    
    return raw_lap





########################################
######## GENERATE FOLDERS ########
########################################


#os.getcwd()
def create_folder(folder_name, construct_token):
    if os.path.exists(folder_name) == False:
        os.mkdir(folder_name)
        print('create : ' + folder_name)
        construct_token += 1
    return construct_token

def generate_folder_structure(sujet):

    construct_token = 0

    os.chdir(path_general)
    
    construct_token = create_folder('Analyses', construct_token)
    construct_token = create_folder('Data', construct_token)
    construct_token = create_folder('Mmap', construct_token)

    #### Analyses
    os.chdir(os.path.join(path_general, 'Analyses'))
    construct_token = create_folder('preprocessing', construct_token)
    construct_token = create_folder('precompute', construct_token)
    construct_token = create_folder('anatomy', construct_token)
    construct_token = create_folder('results', construct_token)
    construct_token = create_folder('protocole', construct_token)
    
        #### preprocessing
    os.chdir(os.path.join(path_general, 'Analyses', 'preprocessing'))
    construct_token = create_folder(sujet, construct_token)
    os.chdir(os.path.join(path_general, 'Analyses', 'preprocessing', sujet))
    construct_token = create_folder('sections', construct_token)
    construct_token = create_folder('info', construct_token)

        #### precompute
    os.chdir(os.path.join(path_general, 'Analyses', 'precompute'))
    construct_token = create_folder(sujet, construct_token)
    construct_token = create_folder('allsujet', construct_token)

            #### allsujet
    os.chdir(os.path.join(path_general, 'Analyses', 'precompute', 'allsujet'))
    construct_token = create_folder('HRV', construct_token)
    construct_token = create_folder('TF', construct_token)
    construct_token = create_folder('PSD_Coh', construct_token)

            #### sujet
    os.chdir(os.path.join(path_general, 'Analyses', 'precompute', sujet))
    construct_token = create_folder('ITPC', construct_token)
    construct_token = create_folder('TF', construct_token)
    construct_token = create_folder('PSD_Coh', construct_token)
    construct_token = create_folder('baselines', construct_token)
    construct_token = create_folder('FC', construct_token)
    construct_token = create_folder('HRV', construct_token)

        #### anatomy
    os.chdir(os.path.join(path_general, 'Analyses', 'anatomy'))
    construct_token = create_folder(sujet, construct_token)

        #### results
    os.chdir(os.path.join(path_general, 'Analyses', 'results'))
    construct_token = create_folder(sujet, construct_token)
    construct_token = create_folder('allplot', construct_token)
    os.chdir(os.path.join(path_general, 'Analyses', 'results', sujet))
    construct_token = create_folder('PSYCHO', construct_token)
    construct_token = create_folder('RESPI', construct_token)
    construct_token = create_folder('TF', construct_token)
    construct_token = create_folder('PSD_Coh', construct_token)
    construct_token = create_folder('ITPC', construct_token)
    construct_token = create_folder('FC', construct_token)
    construct_token = create_folder('HRV', construct_token)
    construct_token = create_folder('df', construct_token)
    construct_token = create_folder('ERP', construct_token)

            #### ERP
    os.chdir(os.path.join(path_general, 'Analyses', 'results', sujet, 'TF'))
    construct_token = create_folder('summary', construct_token)
    construct_token = create_folder('topoplot', construct_token)

            #### TF
    os.chdir(os.path.join(path_general, 'Analyses', 'results', sujet, 'TF'))
    construct_token = create_folder('summary', construct_token)
    construct_token = create_folder('allcond', construct_token)

            #### PSD_Coh
    os.chdir(os.path.join(path_general, 'Analyses', 'results', sujet, 'PSD_Coh'))
    construct_token = create_folder('summary', construct_token)
    construct_token = create_folder('allcond', construct_token)
    construct_token = create_folder('topoplot', construct_token) 

            #### ITPC
    os.chdir(os.path.join(path_general, 'Analyses', 'results', sujet, 'ITPC'))
    construct_token = create_folder('summary', construct_token)
    construct_token = create_folder('allcond', construct_token)
    construct_token = create_folder('topoplot', construct_token)

        #### allplot
    os.chdir(os.path.join(path_general, 'Analyses', 'results', 'allplot'))
    construct_token = create_folder('df', construct_token)
    construct_token = create_folder('TF', construct_token)
    construct_token = create_folder('ITPC', construct_token)
    construct_token = create_folder('FC', construct_token)
    construct_token = create_folder('PSD_Coh', construct_token)

    return construct_token




    


################################################
######## DATA MANAGEMENT CLUSTER ########
################################################


def sync_folders(local_folder, remote_folder, hostname, username, port):

    hostname = '10.69.168.93'
    port = 22
    username = 'jules.granget'

    local_folder = 'N:\\cmo\\Projets\\Olfadys\\NBuonviso2022_jules_olfadys\\EEG_Paris_J\\Script_slurm'
    remote_folder = "/mnt/data/julesgranget/Olfadys/test"

    try:
        # Prompt for the SSH password
        password = getpass.getpass(prompt="Enter your SSH password: ")

        # Build the rsync command
        rsync_command = [
            "rsync", "-avz",  # Flags for archive mode, verbose, and compression
            "-e", f"ssh -p {port}",  # Use SSH for the connection and specify the port
            f"{local_folder}/",  # Local folder (trailing slash ensures content syncing)
            f"{username}@{hostname}:{remote_folder}"  # Remote destination
        ]
        
        print("Running rsync command...")
        # Run the rsync command
        process = subprocess.run(
            rsync_command, 
            input=password,  # Pass the password
            text=True,  # Ensure the password is passed as a string
            capture_output=True  # Capture the output for debugging
        )

        process = subprocess.run(['cd N:\\cmo\\Projets\\Olfadys\\NBuonviso2022_jules_olfadys\\EEG_Paris_J\\Script_Python_EEG_Paris_git'], shell=True, capture_output=True)

        process.stdout
        
        # Check the result
        if process.returncode == 0:
            print("Synchronization successful.")
        else:
            print(f"Rsync failed with error: {process.stderr}")
    
    except Exception as e:
        print(f"An error occurred: {e}")

def ssh_connect(hostname, port, username, password, command):
    """
    Connects to a remote machine using SSH and executes a command.
    
    Parameters:
    - hostname (str): IP address or hostname of the remote machine.
    - port (int): SSH port (default is 22).
    - username (str): SSH username.
    - password (str): SSH password.
    - command (str): Command to execute on the remote machine.
    
    Returns:
    - str: Output of the executed command.
    """
    # Create an SSH client
    ssh_client = paramiko.SSHClient()
    
    # Automatically add the server's SSH key (if not already known)
    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    try:
        # Connect to the remote machine
        print(f"Connecting to {hostname}...")
        ssh_client.connect(hostname, port=port, username=username, password=password)
        print("Connection established.")
        
        # Execute a command
        print(f"Executing command: {command}")
        stdin, stdout, stderr = ssh_client.exec_command(command)
        
        # Read the output and error
        output = stdout.read().decode()
        error = stderr.read().decode()
        
        # Print command output or error
        if output:
            print("Command Output:")
            print(output)
        if error:
            print("Command Error:")
            print(error)
        
        return output if output else error
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    
    finally:
        # Close the connection
        ssh_client.close()
        print("Connection closed.")




def verify_data():

    hostname = '10.69.168.93'
    port = 22
    username = 'jules.granget'
    path_remote_rawdata = "/mnt/data/julesgranget/Olfadys/rawdata"

    origin_path = os.getcwd()

    #### identify data in local
    os.chdir(path_data)

    local_files = {}
    
    for root, _, files in os.walk(path_data):
        for file in files:
            file_path = os.path.join(root, file)
            relative_path = os.path.relpath(file_path, path_data)
            local_files[relative_path] = compute_file_hash(file_path)

    #### identify data in remote
        #### connect
    ssh_client = paramiko.SSHClient()
    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    password = getpass.getpass(prompt="Enter your SSH password: ")
    ssh_client.connect(hostname, port=port, username=username, password=password)
    print(f"Connected to {hostname}")
    sftp = ssh_client.open_sftp()

        #### identify data
    remote_files = {}
            
    def walk_remote_dir(path_remote_rawdata, relative_path=""):

        file_list = sftp.listdir(path_remote_rawdata)
        
        for item in file_list:
            full_path = os.path.join(path_remote_rawdata, item)
            rel_path = os.path.join(relative_path, item)
            
            try:
                if is_remote_directory(sftp, full_path):
                    walk_remote_dir(full_path, rel_path)
                else:
                    remote_files[rel_path] = compute_remote_file_hash(sftp, full_path)
            except IOError:
                continue

    walk_remote_dir(path_remote_rawdata)

    #### compute differences
    differences = {'not_in_remote' : [], 'different_from_local' : [], 'only_in_remote' : []}
        
    for file in local_files:
        if file not in remote_files:
            differences['not_in_remote'].append(file)
        elif local_files[file] != remote_files[file]:
            differences['different_from_local'].append(file)
    
    for file in remote_files:
        if file not in local_files:
            differences['only_in_remote'].append(file)

    #### balance local and remote
    for file in differences['not_in_remote']:

        sftp.put(os.path.join(path_data, file), path_remote_rawdata)

    sftp.chdir(path_remote_rawdata)
    sftp.getcwd()
    
    return is_same, differences






################################
######## SLURM EXECUTE ########
################################


#name_script, name_function, params = 'test', 'slurm_test',  ['Pilote', 2]
def execute_function_in_slurm(name_script, name_function, params):

    python = sys.executable

    #### params to print in script
    params_str = ""
    for params_i in params:
        if isinstance(params_i, str):
            str_i = f"'{params_i}'"
        else:
            str_i = str(params_i)

        if params_i == params[0] :
            params_str = params_str + str_i
        else:
            params_str = params_str + ' , ' + str_i

    #### params to print in script name
    params_str_name = ''
    for params_i in params:

        str_i = str(params_i)

        if params_i == params[0] :
            params_str_name = params_str_name + str_i
        else:
            params_str_name = params_str_name + '_' + str_i
    
    #### script text
    lines = [f'#! {python}']
    lines += ['import sys']
    lines += [f"sys.path.append('{path_main_workdir}')"]
    lines += [f'from {name_script} import {name_function}']
    lines += [f'{name_function}({params_str})']

    cpus_per_task = n_core_slurms
    mem = mem_crnl_cluster
        
    #### write script and execute
    os.chdir(path_slurm)
    slurm_script_name =  f"run_function_{name_function}_{params_str_name}.py" #add params
        
    with open(slurm_script_name, 'w') as f:
        f.writelines('\n'.join(lines))
        os.fchmod(f.fileno(), mode = stat.S_IRWXU)
        f.close()
        
    subprocess.Popen(['sbatch', f'{slurm_script_name}', f'-cpus-per-task={n_core_slurms}', f'-mem={mem_crnl_cluster}']) 

    # wait subprocess to lauch before removing
    #time.sleep(3)
    #os.remove(slurm_script_name)

    print(f'#### slurm submission : from {name_script} execute {name_function}({params})')






#name_script, name_function, params = 'n7_precompute_TF', 'precompute_tf', [cond, session_i, freq_band_list, band_prep_list]
def execute_function_in_slurm_bash(name_script, name_function, params):

    scritp_path = os.getcwd()
    
    python = sys.executable

    #### params to print in script
    params_str = ""
    for i, params_i in enumerate(params):
        if isinstance(params_i, str):
            str_i = f"'{params_i}'"
        else:
            str_i = str(params_i)

        if i == 0 :
            params_str = params_str + str_i
        else:
            params_str = params_str + ' , ' + str_i

    #### params to print in script name
    params_str_name = ''
    for i, params_i in enumerate(params):

        str_i = str(params_i)

        if i == 0 :
            params_str_name = params_str_name + str_i
        else:
            params_str_name = params_str_name + '_' + str_i

    #### remove all txt that block name save
    for txt_remove_i in ["'", "[", "]", "{", "}", ":", " ", ","]:
        if txt_remove_i == " " or txt_remove_i == ",":
            params_str_name = params_str_name.replace(txt_remove_i, '_')
        else:
            params_str_name = params_str_name.replace(txt_remove_i, '')
    
    #### script text
    lines = [f'#! {python}']
    lines += ['import sys']
    lines += [f"sys.path.append('{path_main_workdir}')"]
    lines += [f'from {name_script} import {name_function}']
    lines += [f'{name_function}({params_str})']

    cpus_per_task = n_core_slurms
    mem = mem_crnl_cluster
        
    #### write script and execute
    os.chdir(path_slurm)
    slurm_script_name =  f"run__{name_function}__{params_str_name}.py" #add params
        
    with open(slurm_script_name, 'w') as f:
        f.writelines('\n'.join(lines))
        os.fchmod(f.fileno(), mode = stat.S_IRWXU)
        f.close()
    
    #### script text
    lines = ['#!/bin/bash']
    lines += [f'#SBATCH --job-name={name_function}']
    lines += [f'#SBATCH --output=%slurm_{name_function}_{params_str_name}.log']
    lines += [f'#SBATCH --cpus-per-task={n_core_slurms}']
    lines += [f'#SBATCH --mem={mem_crnl_cluster}']
    lines += [f'srun {python} {os.path.join(path_slurm, slurm_script_name)}']
        
    #### write script and execute
    slurm_bash_script_name =  f"bash__{name_function}__{params_str_name}.batch" #add params
        
    with open(slurm_bash_script_name, 'w') as f:
        f.writelines('\n'.join(lines))
        os.fchmod(f.fileno(), mode = stat.S_IRWXU)
        f.close()

    #### execute bash
    print(f'#### slurm submission : from {name_script} execute {name_function}({params})')
    subprocess.Popen(['sbatch', f'{slurm_bash_script_name}']) 

    # wait subprocess to lauch before removing
    #time.sleep(4)
    #os.remove(slurm_script_name)
    #os.remove(slurm_bash_script_name)

    #### get back to original path
    os.chdir(scritp_path)




#name_script, name_function, params = 'n9_fc_analysis', 'compute_pli_ispc_allband', [sujet]
def execute_function_in_slurm_bash_mem_choice(name_script, name_function, params, mem_required):

    scritp_path = os.getcwd()
    
    python = sys.executable

    #### params to print in script
    params_str = ""
    for i, params_i in enumerate(params):
        if isinstance(params_i, str):
            str_i = f"'{params_i}'"
        else:
            str_i = str(params_i)

        if i == 0 :
            params_str = params_str + str_i
        else:
            params_str = params_str + ' , ' + str_i

    #### params to print in script name
    params_str_name = ''
    for i, params_i in enumerate(params):

        str_i = str(params_i)

        if i == 0 :
            params_str_name = params_str_name + str_i
        else:
            params_str_name = params_str_name + '_' + str_i

    #### remove all txt that block name save
    for txt_remove_i in ["'", "[", "]", "{", "}", ":", " ", ","]:
        if txt_remove_i == " " or txt_remove_i == ",":
            params_str_name = params_str_name.replace(txt_remove_i, '_')
        else:
            params_str_name = params_str_name.replace(txt_remove_i, '')
    
    #### script text
    lines = [f'#! {python}']
    lines += ['import sys']
    lines += [f"sys.path.append('{path_main_workdir}')"]
    lines += [f'from {name_script} import {name_function}']
    lines += [f'{name_function}({params_str})']

    cpus_per_task = n_core_slurms
    mem = mem_crnl_cluster
        
    #### write script and execute
    os.chdir(path_slurm)
    slurm_script_name =  f"run__{name_function}__{params_str_name}.py" #add params
        
    with open(slurm_script_name, 'w') as f:
        f.writelines('\n'.join(lines))
        os.fchmod(f.fileno(), mode = stat.S_IRWXU)
        f.close()
    
    #### script text
    lines = ['#!/bin/bash']
    lines += [f'#SBATCH --job-name={name_function}']
    lines += [f'#SBATCH --output=%slurm_{name_function}_{params_str_name}.log']
    lines += [f'#SBATCH --cpus-per-task={n_core_slurms}']
    lines += [f'#SBATCH --mem={mem_required}']
    lines += [f'srun {python} {os.path.join(path_slurm, slurm_script_name)}']
        
    #### write script and execute
    slurm_bash_script_name =  f"bash__{name_function}__{params_str_name}.batch" #add params
        
    with open(slurm_bash_script_name, 'w') as f:
        f.writelines('\n'.join(lines))
        os.fchmod(f.fileno(), mode = stat.S_IRWXU)
        f.close()

    #### execute bash
    print(f'#### slurm submission : from {name_script} execute {name_function}({params})')
    subprocess.Popen(['sbatch', f'{slurm_bash_script_name}']) 

    # wait subprocess to lauch before removing
    #time.sleep(4)
    #os.remove(slurm_script_name)
    #os.remove(slurm_bash_script_name)

    #### get back to original path
    os.chdir(scritp_path)














################################
######## WAVELETS ########
################################


def get_wavelets():

    #### compute wavelets
    wavelets = np.zeros((nfrex, len(wavetime)), dtype=complex)

    # create Morlet wavelet family
    for fi in range(nfrex):
        
        s = cycles[fi] / (2*np.pi*frex[fi])
        gw = np.exp(-wavetime**2/ (2*s**2)) 
        sw = np.exp(1j*(2*np.pi*frex[fi]*wavetime))
        mw =  gw * sw

        wavelets[fi,:] = mw

    if debug:

        plt.plot(np.sum(np.abs(wavelets),axis=1))
        plt.show()

        plt.pcolormesh(np.real(wavelets))
        plt.show()

        plt.plot(np.real(wavelets)[0,:])
        plt.show()

    return wavelets




def get_wavelets_fc(band_prep, freq):

    #### select wavelet parameters
    if freq[0] < 45:
        wavetime = np.arange(-2,2,1/srate)
        nfrex = nfrex_fc
        ncycle_list = np.linspace(7, 12, nfrex) 

    if freq[0] > 45:
        wavetime = np.arange(-.5,.5,1/srate)
        nfrex = nfrex_fc
        ncycle_list = np.linspace(20, 41, nfrex)

    #### compute wavelets
    frex  = np.linspace(freq[0],freq[1],nfrex)
    wavelets = np.zeros((nfrex,len(wavetime)) ,dtype=complex)

    # create Morlet wavelet family
    for fi in range(0,nfrex):
        
        s = ncycle_list[fi] / (2*np.pi*frex[fi])
        gw = np.exp(-wavetime**2/ (2*s**2)) 
        sw = np.exp(1j*(2*np.pi*frex[fi]*wavetime))
        mw =  gw * sw

        wavelets[fi,:] = mw

    return wavelets





############################
######## LOAD DATA ########
############################


def get_params():

    nwind, nfft, noverlap, hannw = get_params_spectral_analysis(srate)

    params = {'chan_list' : chan_list, 'chan_list_eeg' : chan_list_eeg, 'srate' : srate, 
    'nwind' : nwind, 'nfft' : nfft, 'noverlap' : noverlap, 'hannw' : hannw}

    return params

    

def extract_chanlist_srate_conditions(sujet, monopol):

    path_source = os.getcwd()
    
    #### select conditions to keep
    os.chdir(os.path.join(path_prep, sujet, 'sections'))
    dirlist_subject = os.listdir()

    conditions = []
    for cond in conditions:

        for file in dirlist_subject:

            if file.find(cond) != -1 : 
                conditions.append(cond)
                break

    #### extract data
    band_prep = band_prep_list[0]
    if monopol:
        file_to_search = f'{sujet}_FR_CV_1_{band_prep}.fif'
    else:
        file_to_search = f'{sujet}_FR_CV_1_{band_prep}_bi.fif'

    load_i = []
    for session_i, session_name in enumerate(os.listdir()):
        if ( session_name.find(file_to_search) != -1 ) :
            load_i.append(session_i)
        else:
            continue

    load_name = [os.listdir()[i] for i in load_i][0]

    raw = mne.io.read_raw_fif(load_name, preload=True, verbose='critical')

    srate = int(raw.info['sfreq'])
    chan_list = raw.info['ch_names']
    chan_list_ieeg = chan_list[:-4] # on enlève : nasal, ventral, ECG, ECG_cR

    #### go back to path source
    os.chdir(path_source)

    return conditions, chan_list, chan_list_ieeg, srate


def extract_chanlist_srate_conditions_for_sujet(sujet_tmp, conditions_allsubjects):

    path_source = os.getcwd()
    
    #### select conditions to keep
    os.chdir(os.path.join(path_prep, sujet_tmp, 'sections'))
    dirlist_subject = os.listdir()

    conditions = []
    for cond in conditions_allsubjects:

        for file in dirlist_subject:

            if file.find(cond) != -1 : 
                conditions.append(cond)
                break

    #### extract data
    band_prep = band_prep_list[0]
    cond = conditions[0]

    load_i = []
    for session_i, session_name in enumerate(os.listdir()):
        if ( session_name.find(cond) != -1 ) & ( session_name.find(band_prep) != -1 ):
            load_i.append(session_i)
        else:
            continue

    load_name = [os.listdir()[i] for i in load_i][0]

    raw = mne.io.read_raw_fif(load_name, preload=True, verbose='critical')

    srate = int(raw.info['sfreq'])
    chan_list = raw.info['ch_names']
    chan_list_ieeg = chan_list[:-4] # on enlève : nasal, ventral, ECG, ECG_cR

    #### go back to path source
    os.chdir(path_source)

    return conditions, chan_list, chan_list_ieeg, srate


def load_data_sujet(sujet, cond, odor_i):

    path_source = os.getcwd()
    
    os.chdir(os.path.join(path_prep, sujet, 'sections'))

    raw = mne.io.read_raw_fif(f'{sujet}_{odor_i}_{cond}_wb.fif', preload=True, verbose='critical')

    data = raw.get_data()

    #### go back to path source
    os.chdir(path_source)

    #### free memory
    del raw

    return data



def get_srate(sujet):

    path_source = os.getcwd()
    
    os.chdir(os.path.join(path_prep, sujet, 'sections'))

    raw = mne.io.read_raw_fif(sujet + '_FR_CV_1_lf.fif', preload=True, verbose='critical')
    
    srate = int(raw.info['sfreq'])

    #### go back to path source
    os.chdir(path_source)

    #### free memory
    del raw

    return srate


def get_pos_file(sujet, band_prep):

    path_source = os.getcwd()
    
    os.chdir(os.path.join(path_prep, sujet, 'sections'))

    raw = mne.io.read_raw_fif(f'{sujet}_o_FR_CV_1_{band_prep}.fif', preload=True, verbose='critical')
    
    info = raw.info

    #### go back to path source
    os.chdir(path_source)

    #### free memory
    del raw

    return info



########################################
######## LOAD RESPI FEATURES ########
########################################

def load_respfeatures(sujet):

    path_source = os.getcwd()
    
    os.chdir(os.path.join(path_respfeatures, sujet, 'RESPI'))
    respfeatures_listdir = os.listdir()

    #### remove fig0 and fig1 file
    respfeatures_listdir_clean = []
    for file in respfeatures_listdir :
        if file.find('fig') == -1 :
            respfeatures_listdir_clean.append(file)

    #### get respi features
    respfeatures_allcond = {}

    for cond in conditions:

        respfeatures_allcond[cond] = {}

        for odor_i in odor_list:

            load_i = []
            for session_i, session_name in enumerate(respfeatures_listdir_clean):
                if session_name.find(cond) != -1 and session_name.find(odor_i) != -1:
                    load_i.append(session_i)
                else:
                    continue

            load_list = [respfeatures_listdir_clean[i] for i in load_i]

            respfeatures_allcond[cond][odor_i] = pd.read_excel(load_list[0])

    #### go back to path source
    os.chdir(path_source)

    return respfeatures_allcond



def get_all_respi_ratio(sujet):
    
    respfeatures_allcond = load_respfeatures(sujet)
    
    respi_ratio_allcond = {}

    for cond in conditions:

        if len(respfeatures_allcond[cond]) == 1:

            mean_cycle_duration = np.mean(respfeatures_allcond[cond][0][['insp_duration', 'exp_duration']].values, axis=0)
            mean_inspi_ratio = mean_cycle_duration[0]/mean_cycle_duration.sum()

            respi_ratio_allcond[cond] = [ mean_inspi_ratio ]

        elif len(respfeatures_allcond[cond]) > 1:

            data_to_short = []
            data_to_short_count = 1

            for session_i in range(len(respfeatures_allcond[cond])):   
                
                if session_i == 0 :

                    mean_cycle_duration = np.mean(respfeatures_allcond[cond][session_i][['insp_duration', 'exp_duration']].values, axis=0)
                    mean_inspi_ratio = mean_cycle_duration[0]/mean_cycle_duration.sum()
                    data_to_short = [ mean_inspi_ratio ]

                elif session_i > 0 :

                    mean_cycle_duration = np.mean(respfeatures_allcond[cond][session_i][['insp_duration', 'exp_duration']].values, axis=0)
                    mean_inspi_ratio = mean_cycle_duration[0]/mean_cycle_duration.sum()

                    data_replace = [(data_to_short[0] + mean_inspi_ratio)]
                    data_to_short_count += 1

                    data_to_short = data_replace.copy()
            
            # to put in list
            respi_ratio_allcond[cond] = data_to_short[0] / data_to_short_count

    return respi_ratio_allcond








################################
######## STRETCH ########
################################


#resp_features, data = respfeatures_allcond[cond][odor_i], tf[n_chan,0,:]
def stretch_data(resp_features, nb_point_by_cycle, data, srate):

    #### params
    cycle_times = resp_features[['inspi_time', 'expi_time', 'next_inspi_time']].values
    mean_cycle_duration = np.mean(resp_features[['inspi_duration', 'expi_duration']].values, axis=0)
    # mean_inspi_ratio = mean_cycle_duration[0]/mean_cycle_duration.sum()
    mean_inspi_ratio = 0.5
    times = np.arange(0,data.shape[0])/srate

    #### stretch
    if stretch_TF_auto:

        cycles = physio.deform_traces_to_cycle_template(data.reshape(-1,1), times, cycle_times, points_per_cycle=nb_point_by_cycle, 
                segment_ratios=mean_inspi_ratio, output_mode='stacked')
    else:
        
        cycles = physio.deform_traces_to_cycle_template(data.reshape(-1,1), times, cycle_times, points_per_cycle=nb_point_by_cycle, 
                segment_ratios=ratio_stretch_TF, output_mode='stacked')

    #### clean
    mask = resp_features[resp_features['select'] == 1].index.values
    cycle_clean = cycles[mask, :, :]

    #### reshape
    if np.iscomplex(data[0]):
        data_stretch = np.zeros(( cycle_clean.shape[0], nb_point_by_cycle ), dtype='complex')
    else:
        data_stretch = np.zeros(( cycle_clean.shape[0], nb_point_by_cycle ))

    for cycle_i in range(cycle_clean.shape[0]):

        data_stretch[cycle_i, :] = cycle_clean[cycle_i,:].reshape(-1)

    #### inspect
    if debug == True:

        plt.plot(data_stretch.mean(axis=0))
        plt.show()

    return data_stretch, mean_inspi_ratio





#resp_features, nb_point_by_cycle, data, srate = respfeatures_allcond[cond][odor_i], stretch_point_TF, tf[n_chan,:,:], srate
def stretch_data_tf(resp_features, nb_point_by_cycle, data, srate):

    #### params
    cycle_times = resp_features[['inspi_time', 'expi_time', 'next_inspi_time']].values
    mean_cycle_duration = np.mean(resp_features[['inspi_duration', 'expi_duration']].values, axis=0)
    mean_inspi_ratio = mean_cycle_duration[0]/mean_cycle_duration.sum()
    times = np.arange(0,data.shape[1])/srate

    #### stretch
    if stretch_TF_auto:

        cycles = physio.deform_traces_to_cycle_template(data.T, times, cycle_times, points_per_cycle=nb_point_by_cycle, 
                segment_ratios=mean_inspi_ratio, output_mode='stacked')
    else:
        
        cycles = physio.deform_traces_to_cycle_template(data.T, times, cycle_times, points_per_cycle=nb_point_by_cycle, 
                segment_ratios=ratio_stretch_TF, output_mode='stacked')

    #### clean
    mask = resp_features[resp_features['select'] == 1].index.values
    cycle_clean = cycles[mask, :, :]

    #### reshape
    if np.iscomplex(data[0,0]):
        data_stretch = np.zeros(( cycle_clean.shape[0], data.shape[0], nb_point_by_cycle ), dtype='complex')
    else:
        data_stretch = np.zeros(( cycle_clean.shape[0], data.shape[0], nb_point_by_cycle ))

    for cycle_i in range(cycle_clean.shape[0]):

        data_stretch[cycle_i, :, :] = cycle_clean[cycle_i,:,:].T

    #### inspect
    if debug == True:

        plt.pcolormesh(np.mean(data_stretch, axis=0))
        plt.show()

    return data_stretch, mean_inspi_ratio












########################################
######## LOAD LOCALIZATION ########
########################################



def get_loca_df(sujet, monopol):

    path_source = os.getcwd()

    os.chdir(os.path.join(path_anatomy, sujet))

    if monopol:
        file_plot_select = pd.read_excel(sujet + '_plot_loca.xlsx')
    else:
        file_plot_select = pd.read_excel(sujet + '_plot_loca_bi.xlsx')

    chan_list_ieeg_trc = file_plot_select['plot'][file_plot_select['select'] == 1].values.tolist()

    if sujet[:3] == 'pat':
        chan_list_ieeg_csv = chan_list_ieeg_trc.copy()
    else:
        if monopol:
            chan_list_ieeg_csv, trash = modify_name(chan_list_ieeg_trc)
        else:
            chan_list_ieeg_csv = chan_list_ieeg_trc
        chan_list_ieeg_csv.sort()

    ROI_ieeg = []
    lobes_ieeg = []
    for chan_name in chan_list_ieeg_csv:
        ROI_ieeg.append( file_plot_select['localisation_corrected'].loc[file_plot_select['plot'] == chan_name].values.tolist()[0] )
        lobes_ieeg.append( file_plot_select['lobes_corrected'].loc[file_plot_select['plot'] == chan_name].values.tolist()[0] )

    dict_loca = {'name' : chan_list_ieeg_trc,
                'ROI' : ROI_ieeg,
                'lobes' : lobes_ieeg
                }

    df_loca = pd.DataFrame(dict_loca, columns=dict_loca.keys())

    os.chdir(path_source)

    return df_loca


def get_mni_loca(sujet):

    os.chdir(os.path.join(path_anatomy, sujet))

    file_plot_select = pd.read_excel(sujet + '_plot_loca.xlsx')

    chan_list_txt = open(sujet + '_chanlist_ieeg.txt', 'r')
    chan_list_txt_readlines = chan_list_txt.readlines()
    chan_list_ieeg = [i.replace('\n', '') for i in chan_list_txt_readlines]
    chan_list_ieeg, trash = modify_name(chan_list_ieeg)
    chan_list_ieeg.sort()

    mni_loc = file_plot_select['MNI']

    dict_mni = {}
    for chan_name in chan_list_ieeg:
        mni_nchan = file_plot_select['MNI'].loc[file_plot_select['plot'] == chan_name].values[0]
        mni_nchan = mni_nchan[1:-1]
        mni_nchan_convert = [float(mni_nchan.split(',')[0]), float(mni_nchan.split(',')[1]), float(mni_nchan.split(',')[2])]
        dict_mni[chan_name] = mni_nchan_convert

    return dict_mni










########################################
######## CHANGE NAME CSV TRC ########
########################################

#chan_list = prms['chan_list_ieeg']
def modify_name(chan_list):
    
    chan_list_modified = []
    chan_list_keep = []

    for nchan in chan_list:

        #### what we remove
        if nchan.find("+") != -1:
            continue

        if np.sum([str.isalpha(str_i) for str_i in nchan]) >= 2 and nchan.find('p') == -1:
            continue

        if nchan.find('ECG') != -1:
            continue

        if nchan.find('.') != -1:
            continue

        if nchan.find('*') != -1:
            continue

        #### what we do to chan we keep
        else:

            nchan_mod = nchan.replace(' ', '')
            nchan_mod = nchan_mod.replace("'", 'p')

            if nchan_mod.find('p') != -1:
                split = nchan_mod.split('p')
                letter_chan = split[0]

                if len(split[1]) == 1:
                    num_chan = '0' + split[1] 
                else:
                    num_chan = split[1]

                chan_list_modified.append(letter_chan + 'p' + num_chan)
                chan_list_keep.append(nchan)
                continue

            if nchan_mod.find('p') == -1:
                letter_chan = nchan_mod[0]

                split = nchan_mod[1:]

                if len(split) == 1:
                    num_chan = '0' + split
                else:
                    num_chan = split

                chan_list_modified.append(letter_chan + num_chan)
                chan_list_keep.append(nchan)
                continue


    return chan_list_modified, chan_list_keep












########################################
######## MI ANALYSIS FUNCTIONS ########
########################################



def shuffle_CycleFreq(x):

    cut = int(np.random.randint(low=0, high=len(x), size=1))
    x_cut1 = x[:cut]
    x_cut2 = x[cut:]*-1
    x_shift = np.concatenate((x_cut2, x_cut1), axis=0)

    return x_shift
    

def shuffle_Cxy(x):
   half_size = x.shape[0]//2
   ind = np.random.randint(low=0, high=half_size)
   x_shift = x.copy()
   
   x_shift[ind:ind+half_size] *= -1
   if np.random.rand() >=0.5:
       x_shift *= -1

   return x_shift


def Kullback_Leibler_Distance(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return np.sum(np.where(a != 0, a * np.log(a / b), 0))

def Shannon_Entropy(a):
    a = np.asarray(a, dtype=float)
    return - np.sum(np.where(a != 0, a * np.log(a), 0))

def Modulation_Index(distrib, show=False, verbose=False):
    distrib = np.asarray(distrib, dtype = float)
    
    if verbose:
        if np.sum(distrib) != 1:
            print(f'(!)  The sum of all bins is not 1 (sum = {round(np.sum(distrib), 2)})  (!)')
        
    N = distrib.size
    uniform_distrib = np.ones(N) * (1/N)
    mi = Kullback_Leibler_Distance(distrib, uniform_distrib) / np.log(N)
    
    if show:
        bin_width_deg = 360 / N
        
        doubled_distrib = np.concatenate([distrib,distrib] )
        x = np.arange(0, doubled_distrib.size*bin_width_deg, bin_width_deg)
        fig, ax = plt.subplots(figsize = (8,4))
        
        doubled_uniform_distrib = np.concatenate([uniform_distrib,uniform_distrib] )
        ax.scatter(x, doubled_uniform_distrib, s=2, color='r')
        
        ax.bar(x=x, height=doubled_distrib, width = bin_width_deg/1.1, align = 'edge')
        ax.set_title(f'Modulation Index = {round(mi, 4)}')
        ax.set_xlabel(f'Phase (Deg)')
        ax.set_ylabel(f'Amplitude (Normalized)')
        ax.set_xticks([0,360,720])

    return mi

def Shannon_MI(a):
    a = np.asarray(a, dtype = float)
    N = a.size
    kl_divergence_shannon = np.log(N) - Shannon_Entropy(a)
    return kl_divergence_shannon / np.log(N)



def get_MVL(x):
    _phase = np.arange(0, x.shape[0])*2*np.pi/x.shape[0]
    complex_vec = x*np.exp(1j*_phase) # ici sous la forme du module * angle, r * phi

    MVL = np.abs(np.mean(complex_vec))
    
    if debug:
        fig = plt.figure()
        ax = fig.add_subplot(projection='polar')
        ax.scatter(complex_vec.real, complex_vec.imag)
        ax.scatter(np.mean(complex_vec.real), np.mean(complex_vec.imag), linewidth=3, color='r')
        plt.show()

    return MVL


def get_MI_2sig(x, y):

    #### Freedman and Diaconis rule
    nbins_x = int(np.ceil((x.max() - x.min()) / (2 * scipy.stats.iqr(x)*(x.size**(-1/3)))))
    nbins_y = int(np.ceil((y.max() - y.min()) / (2 * scipy.stats.iqr(y)*(y.size**(-1/3)))))

    #### compute proba
    hist_x = np.histogram(x,bins = nbins_x)[0]
    hist_x = hist_x/np.sum(hist_x)
    hist_y = np.histogram(y,bins = nbins_y)[0]
    hist_y = hist_y/np.sum(hist_y)

    hist_2d = np.histogram2d(x, y, bins=[nbins_x, nbins_y])[0]
    hist_2d = hist_2d / np.sum(hist_2d)

    #### compute MI
    E_x = 0
    E_y = 0
    E_x_y = 0

    for p in hist_x:
        if p!=0 :
            E_x += -p*np.log2(p)

    for p in hist_y:
        if p!=0 :
            E_y += -p*np.log2(p)

    for p0 in hist_2d:
        for p in p0 :
            if p!=0 :
                E_x_y += -p*np.log2(p)

    MI = E_x+E_y-E_x_y

    return MI










########################################
######## SCRIPT ADVANCEMENT ########
########################################


def print_advancement(i, i_final, steps=[25, 50, 75]):

    steps_i = {}
    for step in steps:

        step_i = 0
        while (step_i/i_final*100) < step:
            step_i += 1

        steps_i[step] = step_i

    for step, step_i in steps_i.items():

        if i == step_i:
            print(f'{step}%', flush=True)






################################
######## NORMALIZATION ########
################################


def zscore(x):

    x_zscore = (x - x.mean()) / x.std()

    return x_zscore




def zscore_mat(x):

    _zscore_mat = (x - x.mean(axis=1).reshape(-1,1)) / x.std(axis=1).reshape(-1,1)

    return _zscore_mat



def rscore(x):

    mad = np.median( np.abs(x-np.median(x)) ) # median_absolute_deviation

    rzscore_x = (x-np.median(x)) * 0.6745 / mad

    return rzscore_x
    



def rscore_mat(x):

    mad = np.median(np.abs(x-np.median(x, axis=1).reshape(-1,1)), axis=1) # median_absolute_deviation

    _rscore_mat = (x-np.median(x, axis=1).reshape(-1,1)) * 0.6745 / mad.reshape(-1,1)

    return _rscore_mat






#tf_conv = tf_median_cycle[nchan, :, :]
def norm_tf(sujet, tf_conv, odor_i, norm_method):

    path_source = os.getcwd()

    if norm_method not in ['rscore', 'zscore']:

        #### load baseline
        os.chdir(os.path.join(path_precompute, sujet, 'baselines'))

        baselines = xr.open_dataarray(f'{sujet}_{odor_i}_baselines.nc')

    if norm_method == 'dB':

        for n_chan_i, n_chan in enumerate(chan_list_eeg):

            tf_conv[n_chan_i,:,:] = 10*np.log10(tf_conv[n_chan_i,:,:] / baselines.loc[n_chan, :, 'median'].values.reshape(-1,1))

    if norm_method == 'zscore_baseline':

        for n_chan_i, n_chan in enumerate(chan_list_eeg):

            tf_conv[n_chan_i,:,:] = (tf_conv[n_chan_i,:,:] - baselines.loc[n_chan,:,'mean'].values.reshape(-1,1)) / baselines.loc[n_chan,:,'std'].values.reshape(-1,1)
                
    if norm_method == 'rscore_baseline':

        for n_chan_i, n_chan in enumerate(chan_list_eeg):

            tf_conv[n_chan_i,:,:] = (tf_conv[n_chan_i,:,:] - baselines.loc[n_chan,:,'median'].values.reshape(-1,1)) * 0.6745 / baselines.loc[n_chan,:,'mad'].values.reshape(-1,1)

    if norm_method == 'zscore':

        for n_chan_i, n_chan in enumerate(chan_list_eeg):

            tf_conv[n_chan_i,:,:] = zscore_mat(tf_conv[n_chan_i,:,:])
                
    if norm_method == 'rscore':

        for n_chan_i, n_chan in enumerate(chan_list_eeg):

            tf_conv[n_chan_i,:,:] = rscore_mat(tf_conv[n_chan_i,:,:])


    #### verify baseline
    if debug:

        nchan = 0
        nchan_name = chan_list_eeg[nchan]

        fig, axs = plt.subplots(ncols=2)
        axs[0].set_title('mean std')
        axs[0].plot(baselines.loc[nchan_name,:,'mean'], label='mean')
        axs[0].plot(baselines.loc[nchan_name,:,'std'], label='std')
        axs[0].legend()
        axs[0].set_yscale('log')
        axs[1].set_title('median mad')
        axs[1].plot(baselines.loc[nchan_name,:,'median'], label='median')
        axs[1].plot(baselines.loc[nchan_name,:,'mad'], label='mad')
        axs[1].legend()
        axs[1].set_yscale('log')
        plt.show()

        tf_test = tf_conv[nchan,:,:int(tf_conv.shape[-1]/10)].copy()

        fig, axs = plt.subplots(nrows=6)
        fig.set_figheight(10)
        fig.set_figwidth(15)

        percentile_sel = 0

        vmin = np.percentile(tf_test.reshape(-1),percentile_sel)
        vmax = np.percentile(tf_test.reshape(-1),100-percentile_sel)
        im = axs[0].pcolormesh(tf_test, vmin=vmin, vmax=vmax)
        axs[0].set_title('raw')
        fig.colorbar(im, ax=axs[0])

        tf_baseline = 10*np.log10(tf_test / baselines.loc[chan_list_eeg[nchan], :, 'median'].values.reshape(-1,1))
        vmin = np.percentile(tf_baseline.reshape(-1),percentile_sel)
        vmax = np.percentile(tf_baseline.reshape(-1),100-percentile_sel)
        im = axs[1].pcolormesh(tf_baseline, vmin=vmin, vmax=vmax)
        axs[1].set_title('db')
        fig.colorbar(im, ax=axs[1])

        tf_baseline = (tf_test - baselines.loc[chan_list_eeg[nchan],:,'mean'].values.reshape(-1,1)) / baselines.loc[chan_list_eeg[nchan],:,'std'].values.reshape(-1,1)
        vmin = np.percentile(tf_baseline.reshape(-1),percentile_sel)
        vmax = np.percentile(tf_baseline.reshape(-1),100-percentile_sel)
        im = axs[2].pcolormesh(tf_baseline, vmin=vmin, vmax=vmax)
        axs[2].set_title('zscore')
        fig.colorbar(im, ax=axs[2])

        tf_baseline = (tf_test - baselines.loc[chan_list_eeg[nchan],:,'median'].values.reshape(-1,1)) / baselines.loc[chan_list_eeg[nchan],:,'mad'].values.reshape(-1,1)
        vmin = np.percentile(tf_baseline.reshape(-1),percentile_sel)
        vmax = np.percentile(tf_baseline.reshape(-1),100-percentile_sel)
        im = axs[3].pcolormesh(tf_baseline, vmin=vmin, vmax=vmax)
        axs[3].set_title('rscore')
        fig.colorbar(im, ax=axs[3])

        tf_baseline = zscore_mat(tf_test)
        vmin = np.percentile(tf_baseline.reshape(-1),percentile_sel)
        vmax = np.percentile(tf_baseline.reshape(-1),100-percentile_sel)
        im = axs[4].pcolormesh(tf_baseline, vmin=vmin, vmax=vmax)
        axs[4].set_title('zscore_mat')
        fig.colorbar(im, ax=axs[4])

        tf_baseline = rscore_mat(tf_test)
        vmin = np.percentile(tf_baseline.reshape(-1),percentile_sel)
        vmax = np.percentile(tf_baseline.reshape(-1),100-percentile_sel)
        im = axs[5].pcolormesh(tf_baseline, vmin=vmin, vmax=vmax)
        axs[5].set_title('rscore_mat')
        fig.colorbar(im, ax=axs[5])

        plt.show()

    os.chdir(path_source)

    return tf_conv






########################################
######## HRV ANALYSIS HOMEMADE ########
########################################



#### params
def get_params_hrv_homemade(srate_resample_hrv):
    
    nwind_hrv = int( 128*srate_resample_hrv )
    nfft_hrv = nwind_hrv
    noverlap_hrv = np.round(nwind_hrv/90)
    win_hrv = scipy.signal.windows.hann(nwind_hrv)
    f_RRI = (.1, .5)

    return nwind_hrv, nfft_hrv, noverlap_hrv, win_hrv, f_RRI




#### RRI, IFR
#ecg_i, ecg_cR, srate, srate_resample = ecg_i, ecg_cR, srate, srate_resample_hrv
def get_RRI_IFR(ecg_cR, srate_resample) :

    cR_sec = ecg_cR # cR in sec

    # RRI computation
    RRI = np.diff(cR_sec)
    RRI = np.insert(RRI, 0, np.median(RRI))
    IFR = (1/RRI)

    # interpolate
    f = scipy.interpolate.interp1d(cR_sec, RRI, kind='quadratic', fill_value="extrapolate")
    cR_sec_resample = np.arange(cR_sec[0], cR_sec[-1], 1/srate_resample)
    RRI_resample = f(cR_sec_resample)

    #plt.plot(cR_sec, RRI, label='old')
    #plt.plot(cR_sec_resample, RRI_resample, label='new')
    #plt.legend()
    #plt.show()

    return RRI, RRI_resample, IFR



def get_fig_RRI_IFR(ecg_i, ecg_cR, RRI, IFR, srate, srate_resample):

    cR_sec = ecg_cR # cR in sec
    times = np.arange(0,len(ecg_i))/srate # in sec

    f = scipy.interpolate.interp1d(cR_sec, RRI, kind='quadratic', fill_value="extrapolate")
    cR_sec_resample = np.arange(cR_sec[0], cR_sec[-1], 1/srate_resample)
    RRI_resample = f(cR_sec_resample)

    fig, ax = plt.subplots()
    ax = plt.subplot(411)
    plt.plot(times, ecg_i)
    plt.title('ECG')
    plt.ylabel('a.u.')
    plt.xlabel('s')
    plt.vlines(cR_sec, ymin=min(ecg_i), ymax=max(ecg_i), colors='k')
    plt.subplot(412, sharex=ax)
    plt.plot(cR_sec, RRI)
    plt.title('RRI')
    plt.ylabel('s')
    plt.subplot(413, sharex=ax)
    plt.plot(cR_sec_resample, RRI_resample)
    plt.title('RRI_resampled')
    plt.ylabel('Hz')
    plt.subplot(414, sharex=ax)
    plt.plot(cR_sec, IFR)
    plt.title('IFR')
    plt.ylabel('Hz')
    #plt.show()

    # in this plot one RRI point correspond to the difference value between the precedent RR
    # the first point of RRI is the median for plotting consideration

    return fig




def get_fig_PSD_LF_HF(Pxx, hzPxx, VLF, LF, HF):

    # PLOT
    fig = plt.figure()
    plt.plot(hzPxx,Pxx)
    plt.ylim(0, np.max(Pxx[hzPxx>0.01]))
    plt.xlim([0,.6])
    plt.vlines([VLF, LF, HF], ymin=min(Pxx), ymax=max(Pxx), colors='r')
    #plt.show()
    
    return fig

        
def get_fig_poincarre(RRI):

    RRI_1 = RRI[1:]
    RRI_1 = np.append(RRI_1, RRI[-1]) 

    fig = plt.figure()
    plt.scatter(RRI, RRI_1)
    plt.xlabel('RR (ms)')
    plt.ylabel('RR+1 (ms)')
    plt.title('Poincarré ')
    plt.xlim(.600,1.)
    plt.ylim(.600,1.)

    return fig
    
#### DeltaHR

#RRI, srate_resample, f_RRI, condition = result_struct[keys_result[0]][1], srate_resample, f_RRI, cond 
def get_dHR(RRI_resample, srate_resample, f_RRI):
    
    times = np.arange(0,len(RRI_resample))/srate_resample

        # stairs method
    #RRI_stairs = np.array([])
    #len_cR = len(cR) 
    #for RR in range(len(cR)) :
    #    if RR == 0 :
    #        RRI_i = cR[RR+1]/srate - cR[RR]/srate
    #        RRI_stairs = np.append(RRI_stairs, [RRI_i*1e3 for i in range(int(cR[RR+1]))])
    #    elif RR != 0 and RR != len_cR-1 :
    #        RRI_i = cR[RR+1]/srate - cR[RR]/srate
    #        RRI_stairs = np.append(RRI_stairs, [RRI_i*1e3 for i in range(int(cR[RR+1] - cR[RR]))])
    #    elif RR == len_cR-1 :
    #        RRI_stairs = np.append(RRI_stairs, [RRI_i*1e3 for i in range(int(len(ecg) - cR[RR]))])


    peaks, troughs = find_extrema(RRI_resample, srate_resample, f_RRI)
    peaks_RRI, troughs_RRI = RRI_resample[peaks], RRI_resample[troughs]
    peaks_troughs = np.stack((peaks_RRI, troughs_RRI), axis=1)

    fig_verif = plt.figure()
    plt.plot(times, RRI_resample)
    plt.vlines(peaks/srate_resample, ymin=min(RRI_resample), ymax=max(RRI_resample), colors='b')
    plt.vlines(troughs/srate_resample, ymin=min(RRI_resample), ymax=max(RRI_resample), colors='r')
    #plt.show()

    dHR = np.diff(peaks_troughs/srate_resample, axis=1)*1e3

    fig_dHR = plt.figure()
    ax = plt.subplot(211)
    plt.plot(times, RRI_resample*1e3)
    plt.title('RRI')
    plt.ylabel('ms')
    plt.subplot(212, sharex=ax)
    plt.plot(troughs/srate_resample, dHR)
    plt.hlines(np.median(dHR), xmin=min(times), xmax=max(times), colors='m', label='median = {:.3f}'.format(np.median(dHR)))
    plt.legend()
    plt.title('dHR')
    plt.ylabel('ms')
    plt.vlines(peaks/srate_resample, ymin=0, ymax=0.01, colors='b')
    plt.vlines(troughs/srate_resample, ymin=0, ymax=0.01, colors='r')
    plt.tight_layout()
    #plt.show()

    return fig_verif, fig_dHR

#ecg_allcond[cond][odor_i], ecg_cR_allcond[cond][odor_i], prms_hrv
def ecg_analysis_homemade(ecg_i, srate, srate_resample_hrv, fig_token=False):

    #### load params
    nwind_hrv, nfft_hrv, noverlap_hrv, win_hrv, f_RRI = get_params_hrv_homemade(srate_resample_hrv)

    #### load cR
    ecg_cR = scipy.signal.find_peaks(ecg_i, distance=srate*0.5)[0]
    ecg_cR = ecg_cR/srate

    #### verif
    if debug:
        times = np.arange(ecg_i.shape[0])/srate
        plt.plot(times, ecg_i)
        plt.vlines(ecg_cR, ymin=np.min(ecg_i) ,ymax=np.max(ecg_i), colors='r')
        plt.show()


    #### initiate metrics names
    res_list = ['HRV_MeanNN', 'HRV_SDNN', 'HRV_RMSSD', 'HRV_pNN50', 'HRV_LF', 'HRV_HF', 'HRV_LFHF', 'HRV_SD1', 'HRV_SD2', 'HRV_S', 'HRV_rCOV', 'HRV_MAD', 'HRV_MEDIAN']

    #### RRI
    RRI, RRI_resample, IFR = get_RRI_IFR(ecg_i, ecg_cR, srate, srate_resample_hrv)

    HRV_MeanNN = np.mean(RRI)
    
    #### PSD
    VLF, LF, HF = .04, .15, .4
    AUC_LF, AUC_HF, LF_HF_ratio, hzPxx, Pxx = get_PSD_LF_HF(RRI_resample, srate_resample_hrv, nwind_hrv, nfft_hrv, noverlap_hrv, win_hrv, VLF, LF, HF)

    #### descriptors
    MeanNN, SDNN, RMSSD, NN50, pNN50, COV, mad, median = get_stats_descriptors(RRI)

    #### poincarré
    SD1, SD2, Tot_HRV = get_poincarre(RRI)

    #### df
    res_tmp = [HRV_MeanNN*1e3, SDNN*1e3, RMSSD, pNN50*100, AUC_LF/10, AUC_HF/10, LF_HF_ratio, SD1*1e3, SD2*1e3, Tot_HRV*1e6, COV, mad*1e3, median*1e3]
    data_df = {}
    for i, dv in enumerate(res_list):
        data_df[dv] = [res_tmp[i]]

    hrv_metrics_homemade = pd.DataFrame(data=data_df)

    #### for figures

    #### dHR
    if fig_token:
        fig_verif, fig_dHR = get_dHR(RRI_resample, srate_resample_hrv, f_RRI)

    #### fig
    if fig_token:
        fig_RRI = get_fig_RRI_IFR(ecg_i, ecg_cR, RRI, IFR, srate, srate_resample_hrv)
        fig_PSD = get_fig_PSD_LF_HF(Pxx, hzPxx, VLF, LF, HF) 
        fig_poincarre = get_fig_poincarre(RRI)

        fig_list = [fig_RRI, fig_PSD, fig_poincarre, fig_verif, fig_dHR]

        plt.close('all')

        return hrv_metrics_homemade, fig_list

    else:

        return hrv_metrics_homemade



def get_hrv_metrics_win(RRI):

    #### initiate metrics names
    res_list = ['HRV_MeanNN', 'HRV_SDNN', 'HRV_RMSSD', 'HRV_pNN50', 'HRV_SD1', 'HRV_SD2', 'HRV_S', 'HRV_COV', 'HRV_MAD', 'HRV_MEDIAN']

    HRV_MeanNN = np.mean(RRI)
    
    #### descriptors
    MeanNN, SDNN, RMSSD, NN50, pNN50, COV, mad, median = get_stats_descriptors(RRI)

    #### poincarré
    SD1, SD2, Tot_HRV = get_poincarre(RRI)

    #### df
    res_tmp = [HRV_MeanNN*1e3, SDNN*1e3, RMSSD, pNN50*100, SD1*1e3, SD2*1e3, Tot_HRV*1e6, COV, mad*1e3, median*1e3]
    data_df = {}
    for i, dv in enumerate(res_list):
        data_df[dv] = [res_tmp[i]]

    hrv_metrics_homemade = pd.DataFrame(data=data_df)

    return hrv_metrics_homemade



########################################
######## HRV METRICS HOMEMADE ######## 
########################################



def get_PSD_LF_HF(RRI_resample, prms_hrv, VLF, LF, HF):

    srate_resample, nwind, nfft, noverlap, win = prms_hrv['srate_resample_hrv'], prms_hrv['nwind_hrv'], prms_hrv['nfft_hrv'], prms_hrv['noverlap_hrv'], prms_hrv['win_hrv']

    # DETREND
    RRI_detrend = RRI_resample-np.median(RRI_resample)

    # FFT WELCH
    hzPxx, Pxx = scipy.signal.welch(RRI_detrend, fs=srate_resample, window=win, nperseg=nwind, noverlap=noverlap, nfft=nfft)

    AUC_LF = np.trapz(Pxx[(hzPxx>VLF) & (hzPxx<LF)])
    AUC_HF = np.trapz(Pxx[(hzPxx>LF) & (hzPxx<HF)])
    LF_HF_ratio = AUC_LF/AUC_HF

    return AUC_LF, AUC_HF, LF_HF_ratio, hzPxx, Pxx



def get_stats_descriptors(RRI) :

    MeanNN = np.mean(RRI)

    SDNN = np.std(RRI)

    RMSSD = np.sqrt(np.mean((np.diff(RRI)*1e3)**2))

    NN50 = []
    for RR in range(len(RRI)) :
        if RR == len(RRI)-1 :
            continue
        else :
            NN = abs(RRI[RR+1] - RRI[RR])
            NN50.append(NN)

    NN50 = np.array(NN50)*1e3
    pNN50 = np.sum(NN50>50)/len(NN50)

    mad = np.median( np.abs(RRI-np.median(RRI)) )
    COV = mad / np.median(RRI)

    median = np.median(RRI)

    return MeanNN, SDNN, RMSSD, NN50, pNN50, COV, mad, median


def get_poincarre(RRI):
    RRI_1 = RRI[1:]
    RRI_1 = np.append(RRI_1, RRI[-1]) 

    SD1_val = []
    SD2_val = []
    for RR in range(len(RRI)) :
        if RR == len(RRI)-1 :
            continue
        else :
            SD1_val_tmp = (RRI[RR+1] - RRI[RR])/np.sqrt(2)
            SD2_val_tmp = (RRI[RR+1] + RRI[RR])/np.sqrt(2)
            SD1_val.append(SD1_val_tmp)
            SD2_val.append(SD2_val_tmp)

    SD1 = np.std(SD1_val)
    SD2 = np.std(SD2_val)
    Tot_HRV = SD1*SD2*np.pi

    return SD1, SD2, Tot_HRV



def get_hrv_metrics_homemade(cR_time, prms_hrv, analysis_time='5min'):

    #### get RRI
    cR_sec = cR_time/prms_hrv['srate'] # cR in sec

    if analysis_time == '3min':

        cR_sec_mask = (cR_sec >= 60) & (cR_sec <= 240)
        cR_sec = cR_sec[cR_sec_mask] - 60

    RRI = np.diff(cR_sec)
    RRI = np.insert(RRI, 0, np.median(RRI))

    if debug:
        plt.plot(cR_sec, RRI)
        plt.show()
    
    #### descriptors
    MeanNN, SDNN, RMSSD, NN50, pNN50, COV, mad, median = get_stats_descriptors(RRI)

    #### poincarré
    SD1, SD2, Tot_HRV = get_poincarre(RRI)

    #### PSD
    f = scipy.interpolate.interp1d(cR_sec, RRI, kind='quadratic', fill_value="extrapolate")
    cR_sec_resample = np.arange(cR_sec[0], cR_sec[-1], 1/prms_hrv['srate_resample_hrv'])
    RRI_resample = f(cR_sec_resample)

    if debug:
        plt.plot(cR_sec, RRI, label='raw')
        plt.plot(cR_sec_resample, RRI_resample, label='resampled')
        plt.legend()
        plt.show()

    VLF, LF, HF = .04, .15, .4
    AUC_LF, AUC_HF, LF_HF_ratio, hzPxx, Pxx = get_PSD_LF_HF(RRI_resample, prms_hrv, VLF, LF, HF)

    #### df
    res_tmp = {'HRV_MeanNN' : MeanNN*1e3, 'HRV_SDNN' : SDNN*1e3, 'HRV_RMSSD' : RMSSD, 'HRV_pNN50' : pNN50*100, 'HRV_LF' : AUC_LF/10, 'HRV_HF' : AUC_HF/10, 
               'HRV_LFHF' : LF_HF_ratio, 'HRV_SD1' : SD1*1e3, 'HRV_SD2' : SD2*1e3, 'HRV_S' : Tot_HRV*1e6, 'HRV_COV' : COV, 'HRV_MAD' : mad, 'HRV_MEDIAN' : median}
    
    data_df = {}
    for i, dv in enumerate(prms_hrv['metric_list']):
        data_df[dv] = res_tmp[dv]

    hrv_metrics_homemade = pd.DataFrame([data_df])

    return hrv_metrics_homemade






################################
######## NEUROKIT ######## 
################################

#ecg_i = xr_chunk[sujet_i, cond_i, trial_i, :].data
def nk_analysis(ecg_i, srate):

    ecg_cR = scipy.signal.find_peaks(ecg_i, distance=srate*0.5)[0]
    peaks_dict = {'ECG_R_Peaks' : ecg_cR}
    ecg_peaks = pd.DataFrame(peaks_dict)

    hrv_metrics = nk.hrv(ecg_peaks, sampling_rate=srate, show=False)

    hrv_metrics_name = ['HRV_MeanNN', 'HRV_SDNN', 'HRV_RMSSD', 'HRV_pNN50', 'HRV_LF', 'HRV_HF', 'HRV_LFHF', 'HRV_SD1', 'HRV_SD2', 'HRV_S']

    col_to_drop = []
    col_hrv = list(hrv_metrics.columns.values) 
    for metric_name in col_hrv :
        if (metric_name in hrv_metrics_name) == False :
            col_to_drop.append(metric_name)

    hrv_metrics_short = hrv_metrics.copy()
    hrv_metrics_short = hrv_metrics_short.drop(col_to_drop, axis=1)

    return hrv_metrics_short




########################################
######## PERMUTATION STATS ######## 
########################################


# data_baseline, data_cond = data_baseline_chan, data_cond_chan
def get_permutation_cluster_1d(data_baseline, data_cond, n_surr):

    n_trials_baselines = data_baseline.shape[0]
    n_trials_cond = data_cond.shape[0]
    n_trials_min = np.array([n_trials_baselines, n_trials_cond]).min()

    data_shuffle = np.concatenate((data_baseline, data_cond), axis=0)
    n_trial_tot = data_shuffle.shape[0]

    ttest_vec_shuffle = np.zeros((n_surr, data_cond.shape[-1]))

    pixel_based_distrib = np.zeros((n_surr, 2))

    for surr_i in range(n_surr):

        #### shuffle
        random_sel = np.random.choice(n_trial_tot, size=n_trial_tot, replace=False)
        data_shuffle_baseline = data_shuffle[random_sel[:n_trials_min]]
        data_shuffle_cond = data_shuffle[random_sel[n_trials_min:n_trials_min*2]]

        if debug:
            plt.plot(np.mean(data_shuffle_baseline, axis=0), label='baseline')
            plt.plot(np.mean(data_shuffle_cond, axis=0), label='cond')
            plt.legend()
            plt.show()

            plt.plot(ttest_vec_shuffle[surr_i,:], label='shuffle')
            plt.hlines(0.05, xmin=0, xmax=data_shuffle.shape[-1], color='r')
            plt.legend()
            plt.show()

        #### extract max min
        _min, _max = np.median(data_shuffle_cond, axis=0).min(), np.median(data_shuffle_cond, axis=0).max()
        # _min, _max = np.percentile(np.median(tf_shuffle, axis=0), 1, axis=1), np.percentile(np.median(tf_shuffle, axis=0), 99, axis=1)
        
        pixel_based_distrib[surr_i, 0] = _min
        pixel_based_distrib[surr_i, 1] = _max

    min, max = np.median(pixel_based_distrib[:,0]), np.median(pixel_based_distrib[:,1]) 
    # min, max = np.percentile(pixel_based_distrib[:,0], 50), np.percentile(pixel_based_distrib[:,1], 50)

    if debug:
        plt.plot(np.mean(data_baseline, axis=0), label='baseline')
        plt.plot(np.mean(data_cond, axis=0), label='cond')
        plt.hlines(min, xmin=0, xmax=data_shuffle.shape[-1], color='r', label='min')
        plt.hlines(max, xmin=0, xmax=data_shuffle.shape[-1], color='r', label='max')
        plt.legend()
        plt.show()

    #### thresh data
    data_thresh = np.mean(data_cond, axis=0).copy()

    _mask = np.logical_or(data_thresh < min, data_thresh > max)
    _mask = _mask*1

    if debug:

        plt.plot(_mask)
        plt.show()

    #### thresh cluster
    mask = np.zeros(data_cond.shape[-1])

    _mask[0], _mask[-1] = 0, 0 # to ensure np.diff detection

    if _mask.sum() != 0:
 
        start, stop = np.where(np.diff(_mask) != 0)[0][::2], np.where(np.diff(_mask) != 0)[0][1::2] 
        
        sizes = stop - start
        min_size = np.percentile(sizes, tf_stats_percentile_cluster_manual_perm)
        if min_size < erp_time_cluster_thresh:
            min_size = erp_time_cluster_thresh
        cluster_signi = sizes >= min_size

        mask = np.zeros(data_cond.shape[-1])

        for cluster_i, cluster_p in enumerate(cluster_signi):

            if cluster_p:

                mask[start[cluster_i]:stop[cluster_i]] = 1

    mask = mask.astype('bool')

    if debug:

        plt.plot(mask)
        plt.show()

    return mask





########################################
######## CLUSTER WORKING ######## 
########################################

#name = 'test.png'
def export_fig(name, fig):

    path_pre = os.getcwd()

    os.chdir(path_general)

    fig.savefig(name)

    os.chdir(path_pre)




