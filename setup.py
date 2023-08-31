from setuptools import setup, find_packages


setup(
    name = "TIANA",
    version = "0.1.0",
    author = "Rick Z Li",
    author_email = "zzrickli@gmail.com",
    include_package_data=True,
    packages=find_packages(include=['TIANA', 'TIANA.*']),
    description = ("TIANA is a tool to leverage self-attention to understand motif interactions in genomic data"),
    install_requires=[
        'tensorflow ==2.9.1',
        'numpy ==1.22.4',
        'scipy ==1.9.0',
        'pandas ==1.2.4',
        'matplotlib ==3.3.2',
        'seaborn ==0.12.2',
        'logomaker ==0.8',
        'keras ==2.9.0',
        'sklearn ==0.0.post1',
        'jupyter',

    ],
    setup_requires=['flake8'],
    
    license = "MIT",
    keywords = "deep learning, attention, integrated gradients",
    url = "https://github.com/rzzli/TIANA",
)
