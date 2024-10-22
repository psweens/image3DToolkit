from setuptools import setup, find_packages

setup(
    name="image3DToolkit",
    use_scm_version=True,
    setup_requires=["setuptools_scm"],
    packages=find_packages(),
    install_requires=[
        "numpy",
        "opencv-python",
        "matplotlib",
        "scikit-image",
        "Pillow"
    ],
)
