import setuptools

# exec(open('fastai_tf_fit/version.py').read())

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="keras-utils",
    version="0.0.1",
    author="happiness",
    license="Apache License 2.0",
    description="Fit Tensorflow/Keras models fast",
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords=' deep learning, machine learning, keras, tensorflow',
    url="https://github.com/happiness-ai/tf-fit",
    packages=setuptools.find_packages(),
    # install_requires=['tensorflow'],
    python_requires='==3.6.*',
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.6",
    ],
)
