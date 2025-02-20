from setuptools import setup, find_packages


install_requires = ['yacs', 'mesh_to_sdf==0.0.14',
                    'taichi==1.6.0',
                    'pyglet==1.4.10', 'scipy', 'trimesh', 'gym==0.20.0']

packages = find_packages()
for p in packages:
    assert p == 'doma' or p.startswith('doma.')

setup(name='doma',
      version='1.0.0',
      description='deformable-object-manipulation-suite',
      url='https://github.com/IanYangChina/deformable-object-manipulation',
      author='Xintong Yang',
      author_email='YangX66@cardiff.ac.uk',
      packages=packages,
      package_dir={'deformable-object-manipulation': 'doma'},
      package_data={'doma': [
          'assets/meshes/processed/*.obj',
          'assets/meshes/processed/*.sdf',
          'assets/meshes/raw/*.obj',
          'assets/meshes/voxelized/*.vox',
          'engine/configs/manipulator_cfgs/*.yaml',
      ]},
      classifiers=[
          "Programming Language :: Python :: 3",
          "Operating System :: OS Independent",
      ],
      install_requires=install_requires
      )
