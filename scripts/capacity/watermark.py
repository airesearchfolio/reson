import math, numpy as np, torch

class Watermark:
    def __init__(self,cfg,k):
        self.cfg=cfg; self.k=int(k); self.bank=None
    def build(self):
        arr=np.load(self.cfg.joint_carrier_bank)
        s=self.cfg.watermark.n_identity_carriers
        self.bank=torch.tensor(arr[s:s+self.k],dtype=torch.float32)
        print("Loaded payload carriers:",self.bank.shape)
        return self
    def inject(self,bits,seed,device):
        if len(bits)!=self.k: raise ValueError("Wrong payload length")
        g=torch.Generator(device=device).manual_seed(int(seed))
        z=torch.randn(1,self.cfg.watermark.latent_channels,self.cfg.watermark.latent_h,self.cfg.watermark.latent_w,
                      generator=g,device=device,dtype=torch.float32)
        bank=self.bank.to(device=device,dtype=torch.float32)
        signs=torch.tensor([1.0 if int(b)==1 else -1.0 for b in bits],device=device)[:,None,None,None]
        raw=(signs*bank).sum(0,keepdim=True)
        # IMPORTANT: same constant-total-energy protocol for every K.
        p=raw*(bank[0].norm()/(raw.norm()+1e-12))
        a=float(self.cfg.watermark.alpha_payload)
        return (z+a*p)/math.sqrt(1+a*a)
