class Bio_Genetics_AI:
    """Agent specialized in DNA sequence analysis, Genomics, and Pathology Diagnostics."""
    
    def analyze_dna_mutation(self, reference_dna: str, patient_dna: str):
        """
        Compares a patient's DNA sequence to a reference sequence to find point mutations.
        """
        if len(reference_dna) != len(patient_dna):
            return "Error: Sequences must be of the same length for simple point mutation analysis."
            
        mutations = []
        for i, (ref_base, pat_base) in enumerate(zip(reference_dna, patient_dna)):
            if ref_base != pat_base:
                mutations.append({
                    "position": i + 1,
                    "reference": ref_base,
                    "patient": pat_base
                })
                
        return {
            "total_mutations": len(mutations),
            "mutation_details": mutations,
            "risk_level": "High" if len(mutations) > 3 else "Low"
        }

    def execute_crispr_edit(self, target_dna: str, guide_rna: str, replacement_dna: str):
        """
        Executes a programmatic CRISPR-Cas9 genome edit by verifying the NGG PAM sequence.
        """
        import re
        
        # In real biology, the Cas9 enzyme binds to a specific PAM sequence (NGG)
        # The guide RNA must match the sequence just upstream of the PAM.
        pam_pattern = re.compile(f"({guide_rna})[ACGT]GG")
        match = pam_pattern.search(target_dna)
        
        if not match:
            return {"status": "Failed", "reason": "Valid CRISPR/Cas9 NGG PAM sequence not found downstream of the guide RNA."}
            
        cut_site = match.end(1)
        
        edited_sequence = target_dna[:match.start(1)] + replacement_dna + target_dna[cut_site:]
        return {
            "status": "Success",
            "original_sequence": target_dna,
            "edited_sequence": edited_sequence,
            "edit_start_index": match.start(1),
            "pam_sequence": target_dna[cut_site:cut_site+3]
        }
