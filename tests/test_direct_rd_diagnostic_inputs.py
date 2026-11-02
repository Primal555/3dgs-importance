from pathlib import Path
from types import SimpleNamespace,ModuleType
from unittest.mock import patch

import numpy as np
from PIL import Image
import pytest


def test_selected_loader_needs_no_test_image_files(tmp_path):
    from gaussian_jscc.direct_rd_diagnostic_inputs import selected_train_cameras
    folder=tmp_path/'sparse/0';folder.mkdir(parents=True)
    for name in ('images.bin','cameras.bin'):(folder/name).touch()
    images=tmp_path/'images';images.mkdir()
    extr={i:SimpleNamespace(name=f'frame_{i:03d}.png',camera_id=1,qvec=np.array([1.,0.,0.,0.]),tvec=np.zeros(3)) for i in range(9)}
    for i in range(1,8):Image.new('RGB',(2,2)).save(images/f'frame_{i:03d}.png')
    intr={1:SimpleNamespace(id=1,width=2,height=2,model='PINHOLE',params=np.ones(4))}
    from utils.graphics_utils import BasicPointCloud
    package=ModuleType('scene');package.__path__=[str(Path(__file__).resolve().parents[1]/'scene')]
    gaussian=ModuleType('scene.gaussian_model');gaussian.BasicPointCloud=BasicPointCloud
    # Isolate the unrelated GPU scene initializer; image selection and opening are real.
    with patch.dict('sys.modules',{'scene':package,'scene.gaussian_model':gaussian}), \
         patch('scene.colmap_loader.read_extrinsics_binary',return_value=extr), \
         patch('scene.colmap_loader.read_intrinsics_binary',return_value=intr), \
         patch('utils.camera_utils.cameraList_from_camInfos',side_effect=lambda infos,*_:infos):
        result=selected_train_cameras(tmp_path,['frame_001','frame_003'],2,'images')
        assert [c.image_name for c in result]==['frame_001','frame_003']
        for camera in result:camera.image.close()
        with pytest.raises(ValueError,match='train'):
            selected_train_cameras(tmp_path,['frame_000'],2,'images')
