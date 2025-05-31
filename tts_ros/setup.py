from setuptools import setup

package_name = 'tts_ros'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/model', [
            'config/model.pth',
            'config/vocab.json',
            'config/speakers_xtts.pth',
            'config/config.json'
        ])
            ('share/' + package_name + '/speaker_embeddings/41', glob.glob('config/speaker_embeddings/41/*'))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='your@email.com',
    description='TTS with XTTS inside ROS2',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'tts_node = tts_ros.tts_node:main'
        ],
    },
)
