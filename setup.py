from setuptools import find_packages, setup


setup(
    name="videothinkbench",
    version="0.1.0",
    description="VideoThinkBench unified data generation and evaluation toolkit",
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0",
        "opencv-python>=4.8.0",
        "Pillow>=10.0.0",
        "requests>=2.31.0",
        "tqdm>=4.66.0",
    ],
    packages=find_packages(
        include=[
            "data*",
            "evaluators*",
            "tasks*",
            "training*",
            "utils*",
        ]
    ),
    py_modules=["cli"],
    include_package_data=True,
    package_data={
        "data.visual_puzzles": ["fonts/*", "example_data/*/*.json", "example_data/*/*.png"],
        "evaluators.textcentric": ["config/*.yaml"],
    },
    entry_points={
        "console_scripts": [
            "videothinkbench=cli:main",
        ]
    },
)
