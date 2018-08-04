import time
from fabric.api import env, run
from fabric.context_managers import cd
from fabric.operations import get
from fabric.contrib.files import exists

# Declare remote host and other info
env.hosts = ['login.osgconnect.net']

# Declare remote username and key info (optional)
env.user = 'tsantarra'
# env.password = 'can put password here if not using ssh'
env.key_filename = 'private_ssh_key'


def setup():
    if not exists('Llab-Co'):
        run('git clone https://github.com/tsantarra/Llab-Co')

    with cd('Llab-Co'):
        run('git init')
        run('git reset --hard')
        run('git pull')

        if not exists('Log'):
            run('mkdir -p Log')

        run('chmod +x create_virtual_env.sh')
        run('./create_virtual_env.sh')


def run_tests():
    run('condor_submit osg_setup.submit')


def test_output():
    # Commands to execute on the remote server
    run('rm -R test_dir')
    run("mkdir test_dir")
    with cd('test_dir'):
        run('touch out_test.txt')


def check():
    run('ls')
    run("watch -n2 condor_q " + env.user)


def collect_output():
    # grab files
    get("./out*")


def collect_logs():
    get("./Log*")

# http://www.iac.es/sieinvens/siepedia/pmwiki.php?n=HOWTOs.CondorUsefulCommands <---- YESSS
# watch -n2 condor_q -nobatch   -> live status
# condor_q -better-analyze      -> more detail
# nano error101_job.submit      -> edit file
# condor_q username             -> check status of jobs
