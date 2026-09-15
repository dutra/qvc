"""Rebuild CSC 2.1 counterparts for the full Sep09 pre-cut AGN sample.

Uses the saved TAP response and query in
results/diagnostics/chandra_precut_sep13; does not download new data.
Preserves the original data/cscresults.vot and refuses to overwrite the
rebuilt data/cscresults_precut_sep13.vot. Paths are repository-relative,
independent of the working directory.
"""
from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone
import xml.etree.ElementTree as ET
import h5py
import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord, search_around_sky
from astropy.io.votable import parse

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/diagnostics/chandra_precut_sep13"
SOURCE = ROOT/'results/data/sep09_1033am_chisq_slb_faster_etamodifiedtight_nallexcn8000and2000_iters3disk3_svi10000lr0003w500s250_specaug31w500s250_83cb31d_chisq.h5'
DOWNLOAD = OUT/'csc21_region_download.vot'
DEST = ROOT/'data/cscresults_precut_sep13.vot'
if DEST.exists():
    raise FileExistsError(f'Refusing to overwrite {DEST}')

def read_vot(path):
    v = parse(path).get_first_table()
    t = v.to_table()
    for old, field in zip(t.colnames, v.fields):
        if old != field.name:
            t.rename_column(old, field.name)
    return t

status = [e.attrib.get('value') for e in ET.parse(DOWNLOAD).iter()
          if e.tag.endswith('INFO') and e.attrib.get('name') == 'QUERY_STATUS']
assert status and all(s == 'OK' for s in status), status
with h5py.File(SOURCE) as h:
    d = pd.DataFrame({k: h[k][:] for k in ('ra','dec','z')})
    d.insert(0,'object_id',h['object_id'].asstr()[:])
assert len(d) == 24544 and d.object_id.is_unique
assert np.isfinite(d[['ra','dec']]).all().all()
assert ((d.dec-1/3600 > -1.251) & (d.dec+1/3600 < 1.251)).all()
d.to_csv(OUT/'agn_precut_coordinates.csv',index=False)
t = read_vot(DOWNLOAD)
c = t.to_pandas()
assert c.name.is_unique and np.isfinite(c[['ra','dec']]).all().all()
coords = SkyCoord(d.ra.to_numpy()*u.deg,d.dec.to_numpy()*u.deg)
csc = SkyCoord(c.ra.to_numpy()*u.deg,c.dec.to_numpy()*u.deg)
idx, sep, _ = coords.match_to_catalog_sky(csc)
matched = sep.arcsec < 1.
i_agn, i_csc, pair_sep, _ = search_around_sky(coords,csc,1*u.arcsec)
counts = np.bincount(i_agn[pair_sep.arcsec < 1.],minlength=len(d))
assert np.array_equal(counts > 0,matched)
old = read_vot(ROOT/'data/cscresults.vot').to_pandas()
oldidx, oldsep, _ = coords.match_to_catalog_sky(SkyCoord(old.ra.to_numpy()*u.deg,old.dec.to_numpy()*u.deg))
d['csc_matched'] = matched
d['csc_candidates_within_1arcsec'] = counts
d['csc_nearest_sep_arcsec'] = sep.arcsec
d['old_catalog_matched'] = oldsep.arcsec < 1.
for key in c:
    values = c.iloc[idx][key].reset_index(drop=True)
    d['csc_'+key] = values.where(matched)
d.to_csv(OUT/'all_agn_crossmatch.csv',index=False)
d.loc[matched & ~d.old_catalog_matched].to_csv(OUT/'newly_matched_agn.csv',index=False)
unique_idx = np.unique(idx[matched])
fresh = t[unique_idx]
fresh.meta['description'] = 'CSC 2.1 counterparts within 1 arcsec of the 24544 pre-cut QVC AGNs; nearest counterpart per AGN, deduplicated by CSC source.'
temporary = DEST.with_suffix('.tmp.vot')
fresh.write(temporary,format='votable',overwrite=False)
check = read_vot(temporary)
assert len(check) == len(unique_idx)
check_coords = SkyCoord(check['ra'],check['dec'],unit='deg')
_, check_sep, _ = coords.match_to_catalog_sky(check_coords)
assert np.array_equal(check_sep.arcsec < 1., matched)
temporary.rename(DEST)

summary = []
def summarize(label, subset):
    f = pd.to_numeric(subset.csc_flux_aper_b,errors='coerce')
    return dict(sample=label,n_agn=len(subset),old_matched=int(subset.old_catalog_matched.sum()),
                new_matched=int(subset.csc_matched.sum()),usable_broad_flux=int((np.isfinite(f)&(f>0)).sum()))
summary.append(summarize('all_pre_cut',d))
for tag in ('sep12c','sep13a'):
    base, = (ROOT/'plots/hubble').glob(tag+'*/Flatw0waCDM_joint')
    r = pd.read_csv(base/'hubble_plot_residuals.csv',dtype={'object_id':str})
    q = d.set_index('object_id').loc[r.object_id].reset_index()
    row = summarize(tag+'_selected',q)
    usable = np.isfinite(q.csc_flux_aper_b)&(q.csc_flux_aper_b>0)
    # Exact finite requirements for these saved samples; alphaOX is finite for
    # positive flux and redshift and finite dereddened magnitude.
    valid = usable & np.isfinite(r.m_2500_dereddened) & (q.z>0) & np.isfinite(q.z)
    valid &= np.isfinite(r.residuals) & np.isfinite(r.clipping_sigma)
    row['would_plot_alphaOX'] = int(valid.sum())
    row['would_plot_in_fit_range'] = int((valid & q.z.between(.44,3.16,inclusive='neither')).sum())
    summary.append(row)
pd.DataFrame(summary).to_csv(OUT/'match_counts.csv',index=False)
manifest = dict(created_utc=datetime.now(timezone.utc).isoformat(),catalog_release='CSC 2.1',
                endpoint='https://cda.cfa.harvard.edu/csc21tap/sync',
                query=(OUT/'query.adql').read_text(),match_radius_arcsec=1.,comparison='strictly less',
                selection='All input rows; no Hubble cuts',source=str(SOURCE.relative_to(ROOT)),
                source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                raw_download_sha256=hashlib.sha256(DOWNLOAD.read_bytes()).hexdigest(),
                old_catalog_sha256=hashlib.sha256((ROOT/'data/cscresults.vot').read_bytes()).hexdigest(),
                output=str(DEST.relative_to(ROOT)),output_sha256=hashlib.sha256(DEST.read_bytes()).hexdigest(),
                query_status=status,n_regional_csc=len(t),n_unique_counterparts=len(fresh),
                n_ambiguous_agn=int(sum(counts>1)),counts=summary)
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(manifest,indent=2))
