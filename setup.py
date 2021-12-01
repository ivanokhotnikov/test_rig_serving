import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(name="test_rig",
                 version="0.0.1",
                 author="Ivan Okhotnikov",
                 author_email="ivan.okhotnikov@outlook.com",
                 long_description_content_type="text/markdown",
                 url="https://github.com/ivanokhotnikov/test_rig/",
                 packages=setuptools.find_packages(),
                 classifiers=[
                     "Programming Language :: Python :: 3",
                     "License :: OSI Approved :: MIT License",
                     "Operating System :: OS Independent",
                 ],
                 python_requires='>=3.9',
                 include_package_data=True)
