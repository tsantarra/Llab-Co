import time
from fabric.api import env, run
from fabric.context_managers import cd
from fabric.operations import get

# Declare remote host
env.hosts = ['login.osgconnect.net']

# Declare remote username and key info (optional)
env.user = 'tsantarra'
env.password = ''


def test_run():

    run('git clone https://github.com/tsantarra/Llab-Co')
    with cd('Llab-Co'):
        run("mkdir Log")
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
    run("watch -n2 condor_q " + env.user)


def collect_output():
    # grab files
    with cd('test_dir'):
        get("./out*")


# Commands to execute on the remote server
def OLD_run_demo():
    run("git clone https://github.com/srcole/demo_OSG_python")
    with cd('demo_OSG_python'):
        run("chmod +x create_virtenv.sh")
        run("./create_virtenv.sh")
        run("rm -R python_virtenv_demo")
        run("mv lfp_set/ /stash/user/"+env.user+"/lfp_set/")
        run("tar -cvzf misshapen.tar.gz misshapen")
        run("rm -R misshapen")
        run("mkdir Log")
        run("condor_submit sub_PsTs.submit")
        # Need to wait until done running; should be less than 5 minutes
        time.sleep(300)
        get("./out*")