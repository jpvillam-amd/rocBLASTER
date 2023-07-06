import os
import re
import subprocess
import sys
from pathlib import Path

from setuptools import Extension, setup, find_packages
from setuptools.command.build_ext import build_ext


# A CMakeExtension needs a sourcedir instead of a file list.
# The name must be the _single_ output extension from the CMake build.
# If you need multiple extensions, see scikit-build.
class CMakeExtension(Extension):
    def __init__(self, name: str, sourcedir: str = "") -> None:
        super().__init__(name, sources=[])
        self.sourcedir = os.fspath(Path(sourcedir).resolve())+"/rocBlaster"


class CMakeBuild(build_ext):
    def build_extension(self, ext: CMakeExtension) -> None:
        # Must be in this form due to bug in .resolve() only fixed in Python 3.10+
        ext_fullpath = Path.cwd() / self.get_ext_fullpath(ext.name)
        extdir = ext_fullpath.parent.resolve()

        build_temp = Path(ext.sourcedir) / ext.name
        if not build_temp.exists():
            build_temp.mkdir(parents=True)

        print(f"Build_temp: {build_temp} sourcedir: {ext.sourcedir}")

        env = os.environ.copy()
        env["CXX"] = "/opt/rocm/bin/hipcc"

        subprocess.run(
            ["cmake", ext.sourcedir], cwd=build_temp, check=True, env=env
        )
        subprocess.run(
            ["make"], cwd=build_temp, check=True
        )


setup(
    name="rocBlaster",
    version="0.0.1",
    author="Juan Villamizar",
    author_email="Juan.Villamizar@amd.com",
    description="User tunning wrapper for rocBlas",
    long_description="",
    ext_modules=[CMakeExtension("Tunner")],
    cmdclass={"build_ext": CMakeBuild},
    entry_points = {
        'console_scripts': ['rocBlaster=rocBlaster.command_line:cli'],
    },
    packages = find_packages(),
    package_data = {"":["Tunner/rocBlasFinder.cpython-38-x86_64-linux-gnu.so"]},
    zip_safe=False,
)
