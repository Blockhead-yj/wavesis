import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name="wavesis",
    version="0.0.3",
    author="yjdai",
    author_email="136271877@qq.mail.com",
    description="oop data analysis tools for wave data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="404",
    packages=setuptools.find_packages(),
    classifiers=[],
    python_requires='>=3.7',
)
