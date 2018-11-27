from fabric.api import env, run
from fabric.context_managers import cd
from fabric.operations import get
from fabric.contrib.files import exists

# Declare remote host and other info
env.hosts = ['login.osgconnect.net']

# Declare remote username and key info (optional)
with open('osg_credentials.txt', 'r') as cred_file:
    env.user = cred_file.readline().strip('\n')
    env.password = cred_file.readline().strip('\n')  #''can put password here if not using ssh'
    env.key_filename = cred_file.readline().strip('\n')


def setup():
    if not exists('Llab-Co'):
        run('git clone https://github.com/tsantarra/Llab-Co')

    with cd('Llab-Co'):
        # run('git init')
        # run('git reset --hard')
        run('git pull')

        if not exists('Log'):
            run('mkdir -p Log')


def env():
    run('module --ignore-cache load python/3.7.0')
    with cd('Llab-Co'):
        run('chmod +x create_virtual_env.sh')
        run('./create_virtual_env.sh')


def start():
    run('condor_submit osg_setup.submit')


def check():
    run('ls')
    run('condor_q')
    # run('condor_q -better-analyze')
    # run("watch -n2 condor_q " + env.user)


def get_output():
    run('''files=$(ls out.* 2> /dev/null | wc -l)
           if ls out.* &> /dev/null
           then
                mv out.* out/
           fi''')

    with cd('out'):
        run('if ls *.tar.gz 1> /dev/null 2>&1; then for file in *.tar.gz; do tar -zxf $file; done; fi')
        # run("if ls *.tar.gz 1> /dev/null 2>&1; then find . -name '*.tar.gz' -delete; fi")

    get("./out/Llab-Co/*")


def clear_output():
    run('rm -rf out/*')


def get_logs():
    get("./Log*")


def clear_logs():
    run('rm -rf Log/*')

# http://www.iac.es/sieinvens/siepedia/pmwiki.php?n=HOWTOs.CondorUsefulCommands <---- YESSS
# watch -n2 condor_q -nobatch   -> live status
# condor_q -better-analyze      -> more detail
# nano error101_job.submit      -> edit file
# condor_q username             -> check status of jobs
