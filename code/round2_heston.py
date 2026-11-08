"""Complete the Heston model matrix with unchanged training and cached evaluation."""
from pathlib import Path
from models import Neural
from round2_selection import cached_sampler
import expanded_experiment as ex

def cached_sample(self,n=100000,seed=12345):
 # This driver evaluates a fixed fitted state; no parameter is changed afterward.
 if not hasattr(self,'_round2_draw'):self._round2_draw=cached_sampler(self)
 return self._round2_draw(n,seed)

def main():
 Neural.sample=cached_sample
 ex.OUT=Path(__file__).resolve().parents[1]/'referee_round2'
 ex.synthetic('heston',range(10),['Meta-t','VAE-t','WGAN-i'])
if __name__=='__main__':main()
