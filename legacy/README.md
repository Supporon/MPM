# Legacy Sources

The original implementation remains read-only at:

`../../EarthByte-MPM_Lachlan_Porphyry/`

The two original notebooks stay in that repository for baseline comparison:
`MPM_Porphyry_Lachlan.ipynb` and `MPM_Porphyry_NSW.ipynb`. Phase 1 calls
`lib_mpm.py` as an operator library through `src/features/spatial.py`; it does
not copy or rewrite the Notebook implementation.
