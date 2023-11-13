from pathlib import Path
import yaml
import settings
    

def create_reid_model(reid_model, reid_cfg, device):
    model = None
    if reid_model == 'centroids-reid':
        with open(reid_cfg, "r") as f:
            cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
        from reidd.centroids_reid.reid_model import CentroidsReid
        model = CentroidsReid(cfg, device)
        model.load_model()
    elif reid_model == 'osnet_ain_x1_0':
        # from boxmot.deep.reid_multibackend import ReIDDetectMultiBackend
        # model = ReIDDetectMultiBackend(Path(cfg.get('WEIGHTS')), device)
        with open(reid_cfg, "r") as f:
            cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
        from reidd.osnet.reid_model import OsnetReid
        model = OsnetReid(cfg, device)
    
    elif reid_model in settings.SOLIDER_COLLECTION:
        
        from reidd.soldier.reid_model import SoldierReid
        model = SoldierReid(reid_cfg, device = device)
    
    elif reid_model in settings.CONVNEXT_COLLECTION:
        from reidd.convnext.reid_model import ConvnextNetReid
        model = ConvnextNetReid(reid_cfg, device)
    elif reid_model in settings.NVIDIA_COLLECTION:
        from reidd.nvidia.reid_model import NvidiaReidModel
        model = NvidiaReidModel(reid_cfg, device = device)
    return model
