from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="skin-disease-analyzer",
    version="1.0.0",
    author="Medical AI Team",
    author_email="team@medical-ai.com",
    description="نموذج تحليل أمراض الجلد باستخدام PyTorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/medical-ai/skin-disease-analyzer",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Healthcare Industry",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "jupyter>=1.0.0",
        ],
        "gpu": [
            "torch[cuda]>=1.12.0",
            "torchvision[cuda]>=0.13.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "skin-analyzer-train=training.train:main",
            "skin-analyzer-predict=inference.predict:main",
            "skin-analyzer-export=inference.export_model:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json", "*.txt"],
    },
    zip_safe=False,
)