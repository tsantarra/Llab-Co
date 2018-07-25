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


def test_run():
    if not exists('Llab-Co'):
        run('git clone https://github.com/tsantarra/Llab-Co')

    with cd('Llab-Co'):
        run('git init')
        run('git pull')
        run("mkdir -p Log")
        run("condor_submit osg_setup.submit")

        # Need to wait until done running; should be less than 5 minutes
        time.sleep(300)
        get("./out*")


def setup():
    run('git clone https://github.com/tsantarra/Llab-Co')


def run_demo():
    # Commands to execute on the remote server
    run('rm -R test_dir')
    run("mkdir test_dir")
    with cd('test_dir'):
        run('touch out_test.txt')


def check():
    run('ls')
    #run("watch -n2 condor_q " + env.user)


def collect_output():
    # grab files
    with cd('test_dir'):
        get("./out*")

