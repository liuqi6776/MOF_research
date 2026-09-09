# -*- coding: utf-8 -*-
"""
Agent 2.2: Rational Crystal Engineering & CIF Inverse Modification Engine
Bridges Agent 2.1 Design Rules with PMTransformer Multi-Modal Forward Predictor
"""
import os
import sys
import io
import copy
import numpy as np
import pandas as pd
from ase.io import read, write
from ase import Atom, Atoms


try:
    from scripts.mof_property_predictor import MOFPropertyPredictor
except ImportError:
    from mof_property_predictor import MOFPropertyPredictor


class CIFInverseModifier:
    """
    Rational Inverse Engineering Engine for MOF Crystal Structures
    Takes an existing Seed CIF and performs in silico directed modifications:
    - Functional group grafting (-NH2, -CH3, -CF3, -OH, -Cl, -F)
    - Isomorphic metal node transmetalation (e.g. Mn -> Zn, Cu -> Ni)
    - Automated PoreTuning sieving optimization
    """
    def __init__(self, output_dir: str = "results/generated_cifs"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.predictor = MOFPropertyPredictor()

    def find_seed_cif_path(self, seed_identifier: str) -> str:
        """Finds candidate CIF by MOF name or relative path"""
        clean_name = os.path.basename(seed_identifier).replace(".cif", "")
        if not clean_name.endswith("_clean"):
            clean_name_alt = clean_name + "_clean"
        else:
            clean_name_alt = clean_name.replace("_clean", "")

        candidate_dirs = [
            "PMtransformer/PMTransformer_695GCMC_695(1)/PMTransformer_695GCMC_695/moftransformer_inputs",
            "252_MOF_CIFs",
            "results/generated_cifs"
        ]

        for cdir in candidate_dirs:
            p1 = os.path.join(cdir, f"{clean_name}.cif")
            p2 = os.path.join(cdir, f"{clean_name_alt}.cif")
            if os.path.exists(p1):
                return p1
            if os.path.exists(p2):
                return p2

        if os.path.exists(seed_identifier):
            return seed_identifier

        raise FileNotFoundError(f"Seed CIF structure '{seed_identifier}' not found in known repositories.")

    def graft_functional_group(
        self,
        seed_cif: str,
        group_type: str = "-NH2",
        substitution_ratio: float = 0.33,
        out_filename: str = None
    ):
        """
        Grafts functional groups onto organic linker hydrogens
        group_type: '-NH2' (Polar), '-CH3' (Hydrophobic CALF-20), '-CF3' (Polarized), '-OH', '-Cl', '-F'
        """
        cif_path = self.find_seed_cif_path(seed_cif)
        atoms = read(cif_path)
        base_name = os.path.basename(cif_path).replace(".cif", "")

        positions = atoms.get_positions()
        symbols = np.array(atoms.get_chemical_symbols())
        h_indices = np.where(symbols == "H")[0]
        c_indices = np.where(symbols == "C")[0]

        if len(c_indices) == 0:
            raise ValueError(f"CIF '{base_name}' lacks carbon linker backbone for functionalization.")

        new_atoms = atoms.copy()
        rng = np.random.RandomState(42)

        if len(h_indices) > 0:
            # Mode A: Direct C-H substitution
            n_sub = max(1, int(len(h_indices) * substitution_ratio))
            target_h = rng.choice(h_indices, size=n_sub, replace=False)

            for h_idx in target_h:
                dists = new_atoms.get_distances(h_idx, c_indices, mic=True)
                closest_c = c_indices[np.argmin(dists)]
                vec = new_atoms.get_distance(closest_c, h_idx, mic=True, vector=True)
                u = vec / (np.linalg.norm(vec) + 1e-6)
                c_pos = positions[closest_c]

                ortho = np.array([-u[1], u[0], 0.0])
                if np.linalg.norm(ortho) < 1e-3:
                    ortho = np.array([0.0, -u[2], u[1]])
                ortho = ortho / (np.linalg.norm(ortho) + 1e-6)

                if group_type == "-NH2":
                    n_pos = c_pos + 1.40 * u
                    new_atoms[h_idx].symbol = "N"
                    new_atoms[h_idx].position = n_pos
                    new_atoms.append(Atom("H", position=n_pos + 0.82 * u + 0.58 * ortho))
                    new_atoms.append(Atom("H", position=n_pos + 0.82 * u - 0.58 * ortho))
                elif group_type == "-CH3":
                    c_me = c_pos + 1.50 * u
                    new_atoms[h_idx].symbol = "C"
                    new_atoms[h_idx].position = c_me
                    new_atoms.append(Atom("H", position=c_me + 0.85 * u + 0.60 * ortho))
                    new_atoms.append(Atom("H", position=c_me + 0.85 * u - 0.60 * ortho))
                    ortho2 = np.cross(u, ortho)
                    new_atoms.append(Atom("H", position=c_me + 0.85 * u + 0.60 * ortho2))
                elif group_type == "-CF3":
                    c_cf3 = c_pos + 1.50 * u
                    new_atoms[h_idx].symbol = "C"
                    new_atoms[h_idx].position = c_cf3
                    new_atoms.append(Atom("F", position=c_cf3 + 0.95 * u + 0.70 * ortho))
                    new_atoms.append(Atom("F", position=c_cf3 + 0.95 * u - 0.70 * ortho))
                    ortho2 = np.cross(u, ortho)
                    new_atoms.append(Atom("F", position=c_cf3 + 0.95 * u + 0.70 * ortho2))
                elif group_type == "-OH":
                    o_pos = c_pos + 1.36 * u
                    new_atoms[h_idx].symbol = "O"
                    new_atoms[h_idx].position = o_pos
                    new_atoms.append(Atom("H", position=o_pos + 0.80 * u + 0.55 * ortho))
                elif group_type in ["-Cl", "-F"]:
                    hal = group_type.replace("-", "")
                    d_bond = 1.75 if hal == "Cl" else 1.35
                    new_atoms[h_idx].symbol = hal
                    new_atoms[h_idx].position = c_pos + d_bond * u
        else:
            # Mode B: Geometric grafting onto unsaturated linker carbons (for CIFs lacking explicit H)
            n_sub = max(1, int(len(c_indices) * min(0.30, substitution_ratio)))
            target_c = rng.choice(c_indices, size=n_sub, replace=False)

            for c_idx in target_c:
                c_pos = positions[c_idx]
                dists = new_atoms.get_distances(c_idx, range(len(new_atoms)), mic=True)
                nbr_indices = [idx for idx in np.argsort(dists)[1:4] if dists[idx] < 1.65]
                if len(nbr_indices) >= 2:
                    v1 = new_atoms.get_distance(c_idx, nbr_indices[0], mic=True, vector=True)
                    v2 = new_atoms.get_distance(c_idx, nbr_indices[1], mic=True, vector=True)
                    bisect = -(v1 / (np.linalg.norm(v1) + 1e-6) + v2 / (np.linalg.norm(v2) + 1e-6))
                    norm_b = np.linalg.norm(bisect)
                    u = bisect / norm_b if norm_b > 1e-3 else np.array([0.0, 0.0, 1.0])
                else:
                    u = np.array([0.0, 0.0, 1.0])

                ortho = np.array([-u[1], u[0], 0.0])
                if np.linalg.norm(ortho) < 1e-3:
                    ortho = np.array([0.0, -u[2], u[1]])
                ortho = ortho / (np.linalg.norm(ortho) + 1e-6)

                if group_type == "-NH2":
                    n_pos = c_pos + 1.40 * u
                    new_atoms.append(Atom("N", position=n_pos))
                    new_atoms.append(Atom("H", position=n_pos + 0.82 * u + 0.58 * ortho))
                    new_atoms.append(Atom("H", position=n_pos + 0.82 * u - 0.58 * ortho))
                elif group_type == "-CH3":
                    c_me = c_pos + 1.50 * u
                    new_atoms.append(Atom("C", position=c_me))
                    new_atoms.append(Atom("H", position=c_me + 0.85 * u + 0.60 * ortho))
                    new_atoms.append(Atom("H", position=c_me + 0.85 * u - 0.60 * ortho))
                elif group_type == "-CF3":
                    c_cf3 = c_pos + 1.50 * u
                    new_atoms.append(Atom("C", position=c_cf3))
                    new_atoms.append(Atom("F", position=c_cf3 + 0.95 * u + 0.70 * ortho))
                    new_atoms.append(Atom("F", position=c_cf3 + 0.95 * u - 0.70 * ortho))
                elif group_type == "-OH":
                    o_pos = c_pos + 1.36 * u
                    new_atoms.append(Atom("O", position=o_pos))
                    new_atoms.append(Atom("H", position=o_pos + 0.80 * u + 0.55 * ortho))
                elif group_type in ["-Cl", "-F"]:
                    hal = group_type.replace("-", "")
                    d_bond = 1.75 if hal == "Cl" else 1.35
                    new_atoms.append(Atom(hal, position=c_pos + d_bond * u))

        new_atoms.wrap()

        tag = group_type.replace("-", "")
        if out_filename is None:
            out_filename = f"{base_name}_opt_grafted_{tag}.cif"
        out_path = os.path.join(self.output_dir, out_filename)
        write(out_path, new_atoms)
        return out_path, {
            "action": f"Grafted {group_type}",
            "substitution_ratio": substitution_ratio,
            "modified_atoms_count": n_sub,
            "total_atoms_after": len(new_atoms),
            "new_formula": new_atoms.get_chemical_formula()
        }

    def swap_metal_node(
        self,
        seed_cif: str,
        new_metal: str = "Zn",
        out_filename: str = None
    ):
        """
        Performs isomorphic transmetalation on inorganic SBU metal nodes
        new_metal: 'Zn', 'Cu', 'Ni', 'Co', 'Cd', 'Mg', 'Mn'
        """
        cif_path = self.find_seed_cif_path(seed_cif)
        atoms = read(cif_path)
        base_name = os.path.basename(cif_path).replace(".cif", "")

        known_metals = {"Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Zr", "Cd", "Mg", "Al", "Cr", "V", "Mo", "Ru"}
        symbols = atoms.get_chemical_symbols()
        curr_metals = list(set(symbols).intersection(known_metals))

        if not curr_metals:
            raise ValueError(f"No recognizable transition metal found in CIF '{base_name}'.")

        old_metal = curr_metals[0]
        new_symbols = [new_metal if s == old_metal else s for s in symbols]
        new_atoms = atoms.copy()
        new_atoms.set_chemical_symbols(new_symbols)

        if out_filename is None:
            out_filename = f"{base_name}_swap_{old_metal}_to_{new_metal}.cif"
        out_path = os.path.join(self.output_dir, out_filename)
        write(out_path, new_atoms)
        return out_path, {
            "action": f"Metal node swap {old_metal} -> {new_metal}",
            "replaced_metal": old_metal,
            "new_metal": new_metal,
            "swapped_atoms_count": symbols.count(old_metal),
            "new_formula": new_atoms.get_chemical_formula()
        }

    def optimize_and_evaluate(
        self,
        seed_cif: str,
        strategy: str = "PoreTuning_Amination",
        target_metal: str = "Zn"
    ) -> dict:
        """
        Full End-to-End Rational Optimization Pipeline:
        Seed CIF -> Agent 2.1 Decision -> Structural Modification -> Forward PMTransformer Evaluation
        """
        cif_orig = self.find_seed_cif_path(seed_cif)
        base_name = os.path.basename(cif_orig).replace(".cif", "")
        
        # 1. Baseline Evaluation
        pred_orig = self.predictor.predict_properties(cif_orig)
        p_orig = pred_orig["predictions"]
        
        # 2. Execute Directed CIF Modification
        mod_log = {}
        if strategy == "PoreTuning_Amination":
            out_cif, mod_log = self.graft_functional_group(cif_orig, group_type="-NH2", substitution_ratio=0.35)
        elif strategy == "Hydrophobic_CALF20_Methylation":
            out_cif, mod_log = self.graft_functional_group(cif_orig, group_type="-CH3", substitution_ratio=0.35)
        elif strategy == "Fluorinated_SIFSIX_Polarization":
            out_cif, mod_log = self.graft_functional_group(cif_orig, group_type="-CF3", substitution_ratio=0.30)
        elif strategy == "Metal_Node_Transmetalation":
            out_cif, mod_log = self.swap_metal_node(cif_orig, new_metal=target_metal)
        elif strategy == "Dual_Synergy_Metal_and_Linker":
            cif_metal, log_m = self.swap_metal_node(cif_orig, new_metal=target_metal, out_filename=f"{base_name}_temp_metal.cif")
            out_cif, mod_log = self.graft_functional_group(cif_metal, group_type="-NH2", substitution_ratio=0.35, out_filename=f"{base_name}_dual_opt_{target_metal}_NH2.cif")
            mod_log["metal_log"] = log_m
            if os.path.exists(cif_metal):
                os.remove(cif_metal)
        else:
            raise ValueError(f"Unknown optimization strategy '{strategy}'.")

        # 3. Post-Modification PMTransformer Forward Evaluation
        pred_opt = self.predictor.predict_properties(out_cif)
        p_opt = pred_opt["predictions"]

        # 4. Synthesize Improvement Metrics
        comparison = {}
        key_props = [
            ("selectivity_real", "CO2/N2 Actual Selectivity", "Higher is better"),
            ("co2_015bar", "CO2 Uptake @ 0.15bar Flue (mol/kg)", "Higher is better"),
            ("co2_1bar", "CO2 Uptake @ 1.0bar 1atm (mol/kg)", "Higher is better"),
            ("qst_widom", "CO2 Adsorption Heat Qst (kJ/mol)", "Optimal [25-32 kJ/mol]"),
            ("vsa_working_capacity", "VSA Working Capacity (mol/kg)", "Higher is better"),
            ("pld_pred", "Pore Limiting Diameter PLD (Å)", "Target [3.3-5.0 Å]"),
            ("pe_vsa", "Parasitic Energy VSA (kJ/mol)", "Lower is better")
        ]

        for k, lbl, goal in key_props:
            v_orig = p_orig.get(k, {}).get("value", 0.0)
            v_opt = p_opt.get(k, {}).get("value", 0.0)
            delta = round(v_opt - v_orig, 2)
            pct = round((delta / (v_orig + 1e-6)) * 100, 1)
            comparison[k] = {
                "label": lbl,
                "goal": goal,
                "original": v_orig,
                "optimized": v_opt,
                "delta": delta,
                "percent_change": pct
            }

        return {
            "seed_cif": cif_orig,
            "generated_cif": out_path if "out_path" in locals() else out_cif,
            "strategy": strategy,
            "modification_details": mod_log,
            "comparison": comparison,
            "summary_verdict": f"Optimization via {strategy} successfully generated new periodic crystal '{os.path.basename(out_cif)}'."
        }


if __name__ == "__main__":
    modifier = CIFInverseModifier()
    res = modifier.optimize_and_evaluate(
        seed_cif="BEWWEH_clean",
        strategy="PoreTuning_Amination"
    )
    print("\n" + "="*80)
    print("Agent 2.2 Rational Crystal Inverse Optimization Results:")
    print("="*80)
    print(f"Seed CIF     : {res['seed_cif']}")
    print(f"Generated CIF: {res['generated_cif']}")
    print(f"Strategy     : {res['strategy']}")
    print("-" * 80)
    for k, item in res["comparison"].items():
        print(f"{item['label']:<36}: Orig={item['original']:<6.2f} -> Opt={item['optimized']:<6.2f} (Delta={item['delta']:+6.2f}, {item['percent_change']:+6.1f}%)")
    print("="*80)
