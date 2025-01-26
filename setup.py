import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='S2Generator',  # 使用包的名称
    packages=setuptools.find_packages(),
    version='0.0.1',  # 包的版本号，应遵循语义版本控制规则
    description='A series-symbol (S2) dual-modality data generation mechanism, enabling the unrestricted creation of high-quality time series data paired with corresponding symbolic representations.',  # 包的简短描述
    url='https://github.com/wwhenxuan/S2Generator',  # 项目的地址通常来说是github
    author='whenxuan',
    author_email='wwhenxuan@gmail.com',
    keywords=['Time Series', 'Data Generation'],  # 在PyPI上搜索的相应的关键词
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        'Development Status :: 3 - Alpha',  # Alpha表示当前包并不稳定
        'Intended Audience :: Science/Research',  # 当前包使用的人群这里是科研和研究人员
        'Topic :: Scientific/Engineering :: Mathematics',  # 应用的领域
        'Topic :: Scientific/Engineering :: Artificial Intelligence',  # 应用的领域
        'License :: OSI Approved :: MIT License',  # 使用的执照
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    python_requires='>=3.9',  # 最低的Python版本限制
    install_requires=[
        'numpy==1.24.4',
        'scipy==1.14.1',
        'matplotlib==3.9.2'
    ]  # 手动指定依赖的Python以及最低的版本
)