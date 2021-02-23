from os.path import join
from waflib.extras.test_base import summary
from waflib.extras.symwaf2ic import get_toplevel_path


def depends(ctx):
    ctx("code-format")


def options(opt):
    opt.load("test_base")
    opt.load("pytest")


def configure(conf):
    conf.load("test_base")
    conf.load("pytest")
    conf.load('python')
    conf.check_python_version()


def build(bld):
    bld(name="timedatasets-libs",
        features="py use pylint pycodestyle",
        source=bld.path.ant_glob("src/**/*.py"),
        relative_trick=True,
        install_path="${PREFIX}/lib",
        install_from="src",
        pylint_config=join(get_toplevel_path(), "code-format", "pylintrc"),
        pycodestyle_config=join(get_toplevel_path(), "code-format", "pycodestyle"))

    bld(name="timedatasets-tests",
        tests=bld.path.ant_glob("tests/**/*.py"),
        features="use pytest pylint pycodestyle",
        use="timedatasets-libs",
        install_path="${PREFIX}/bin/tests",
        pylint_config=join(get_toplevel_path(), "code-format", "pylintrc"),
        pycodestyle_config=join(get_toplevel_path(), "code-format", "pycodestyle"))

    bld.add_post_fun(summary)
