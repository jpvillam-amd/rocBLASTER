import os
import re
import subprocess
import sys
from pathlib import Path

# import glob
import shutil

from setuptools import Extension, setup, find_packages
from setuptools.command.build_ext import build_ext
from setuptools.command.install_lib import install_lib
from distutils.command.install_data import install_data


# A CMakeExtension needs a sourcedir instead of a file list.
# The name must be the _single_ output extension from the CMake build.
# If you need multiple extensions, see scikit-build.
class CMakeExtension(Extension):
    def __init__(self, name: str, sourcedir: str = "") -> None:
        super().__init__(name, sources=[])
        self.sourcedir = os.fspath(Path(sourcedir).resolve()) + "/rocBlaster"


class CMakeBuild(build_ext):
    def build_extension(self, ext: CMakeExtension) -> None:
        # Must be in this form due to bug in .resolve() only fixed in Python 3.10+
        ext_fullpath = Path.cwd() / self.get_ext_fullpath(ext.name)
        extdir = ext_fullpath.parent.resolve()

        build_temp = Path(ext.sourcedir) / ext.name
        if not build_temp.exists():
            build_temp.mkdir(parents=True)

        print(f"Build_temp: {build_temp} sourcedir: {ext.sourcedir}")
        self.distribution.bin_dir = build_temp

        env = os.environ.copy()
        env["CXX"] = "/opt/rocm/bin/hipcc"

        subprocess.run(["cmake", "-DCMAKE_PREFIX_PATH=/opt/rocm", ext.sourcedir], cwd=build_temp, check=True, env=env)
        subprocess.run(["make"], cwd=build_temp, check=True)
        # so_files = glob.glob(f"{build_temp}/*.so")
        # print(f"found {so_files} from {build_temp}/*")
        # for so_file in so_files:
        #    print(f"Moving {so_file} to {ext.sourcedir}")
        #    subprocess.run(["mv", so_file, ext.sourcedir], cwd=build_temp, check=True)


class InstallCMakeLibsData(install_data):
    """
    Just a wrapper to get the install data into the egg-info

    Listing the installed files in the egg-info guarantees that
    all of the package files will be uninstalled when the user
    uninstalls your package through pip
    """

    def run(self):
        """
        Outfiles are the libraries that were built using cmake
        """

        # There seems to be no other way to do this; I tried listing the
        # libraries during the execution of the InstallCMakeLibs.run() but
        # setuptools never tracked them, seems like setuptools wants to
        # track the libraries through package data more than anything...
        # help would be appriciated

        # TODO: This seems to be coping to a top level, needs fixing.

        self.outfiles = self.distribution.data_files


class InstallCMakeLibs(install_lib):
    """
    Get the libraries from the parent distribution, use those as the outfiles

    Skip building anything; everything is already built, forward libraries to
    the installation step
    """

    def run(self):
        """
        Copy libraries from the bin directory and place them as appropriate
        """

        self.announce("Moving library files", level=3)

        # We have already built the libraries in the previous build_ext step

        self.skip_build = True
        print(f"JUAN: {self.distribution} {self.build_dir} {dir(self.distribution)}")

        bin_dir = self.distribution.bin_dir

        # Depending on the files that are generated from your cmake
        # build chain, you may need to change the below code, such that
        # your files are moved to the appropriate location when the installation
        # is run

        libs = [
            os.path.join(bin_dir, _lib)
            for _lib in os.listdir(bin_dir)
            if os.path.isfile(os.path.join(bin_dir, _lib))
            and os.path.splitext(_lib)[1] in [".dll", ".so"]
        ]
        print(f"JUAN FOUND LIBS: {libs}")

        for lib in libs:
            shutil.move(lib, os.path.join(self.build_dir, os.path.basename(lib)))

        # Mark the libs for installation, adding them to
        # distribution.data_files seems to ensure that setuptools' record
        # writer appends them to installed-files.txt in the package's egg-info
        #
        # Also tried adding the libraries to the distribution.libraries list,
        # but that never seemed to add them to the installed-files.txt in the
        # egg-info, and the online recommendation seems to be adding libraries
        # into eager_resources in the call to setup(), which I think puts them
        # in data_files anyways.
        #
        # What is the best way?

        # These are the additional installation files that should be
        # included in the package, but are resultant of the cmake build
        # step; depending on the files that are generated from your cmake
        # build chain, you may need to modify the below code

        self.distribution.data_files = [
            os.path.join(self.install_dir, os.path.basename(lib)) for lib in libs
        ]

        # Must be forced to run after adding the libs to data_files

        self.distribution.run_command("install_data")

        super().run()


setup(
    name="rocBlaster",
    version="0.0.1",
    author="Juan Villamizar",
    author_email="Juan.Villamizar@amd.com",
    description="User tunning wrapper for rocBlas",
    long_description="",
    ext_modules=[CMakeExtension("Tunner")],
    cmdclass={
        "build_ext": CMakeBuild,
        "install_lib": InstallCMakeLibs,
        "install_data": InstallCMakeLibsData,
    },
    entry_points={
        "console_scripts": ["rocBlaster=rocBlaster.command_line:cli"],
    },
    include_package_data=True,
    packages=["rocBlaster"],
    package_dir={"rocBlaster": "rocBlaster/"},
    package_data={"rocBlaster": ["*.so"]},
    zip_safe=False,
)
