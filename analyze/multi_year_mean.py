import xarray as xr
import re
from pathlib import Path
import numpy as np
base = Path('/data/hjj/SEJ/data/AVISO_0.125deg_indian_ocean')

datapath = sorted(
    (p for p in base.iterdir() if p.suffix == '.nc' and re.search(r'\d+', p.stem)),
    key=lambda p: int(re.search(r'\d+', p.stem).group())
)

time_span = slice('1993-01-01', '2021-12-31')

data = xr.open_mfdataset(datapath, combine='by_coords').sel(time=time_span)

adt_clim = data.adt.mean('time', skipna=True)
np.save(base/Path('adt_clim.npy'), adt_clim.values.astype(np.float32))
print(adt_clim.shape)
