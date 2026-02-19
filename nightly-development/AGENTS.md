# AGENTS.md - PyMultiWFN Ralph Loop

## Test Commands (Backpressure)

```bash
# Run bonding tests
pytest tests/analysis/test_bonding.py -v

# Run with coverage
pytest tests/analysis/test_bonding.py --cov=pymultiwfn --cov-report=html

# Run all tests
pytest tests/ -v

# Run specific test
pytest tests/analysis/test_bonding.py::TestIntegration::test_mayer_vs_wiberg -v
```

## Project Structure

```
pymultiwfn/
├── io/
│   └── parsers/
│       └── wfn.py           # WFN parser (needs overlap matrix)
├── integrals.py             # Integral calculation functions
├── analysis/
│   ├── bonding.py           # Bond order calculations
│   └── population.py        # Population analysis
tests/
├── analysis/
│   └── test_bonding.py      # Bonding tests
```

## Key Functions

### overlap_gaussian_primitive()
Calculate overlap between two Gaussian primitives.

Parameters:
- exp1, exp2: Exponents
- coord1, coord2: Coordinates (x, y, z)
- l1, m1, n1: Angular momentum for primitive 1
- l2, m2, n2: Angular momentum for primitive 2

Returns: Overlap integral value

### WFN Parser
Located in `pymultiwfn/io/parsers/wfn.py`

Current issue: Sets `wavefunction.overlap = np.eye(num_basis)`

Need: Calculate actual overlap matrix from basis set info

## Failing Tests (Before Implementation)

```
tests/analysis/test_bonding.py::TestIntegration::test_mayer_vs_wiberg
- Issue: Wiberg != Mayer (8.2% difference)
- Cause: Overlap matrix is identity

tests/analysis/test_bonding.py::TestParameterized::test_bond_orders_in_range[h2_wavefunction]
- Issue: H-H bond order ~0.005 vs expected ~1.0
- Cause: Overlap matrix is identity

tests/analysis/test_bonding.py::TestParameterized::test_bond_orders_in_range[c2h2_wavefunction]
- Issue: C≡C bond order 3.697 vs expected 2.5-3.5
- Cause: Overlap matrix is identity
```

## Success Criteria

1. ✅ `calculate_overlap_matrix()` function implemented
2. ✅ WFN parser uses calculated overlap matrix
3. ✅ All bonding tests pass
4. ✅ Test coverage >= 80%
5. ✅ Code follows PEP 8
6. ✅ Documentation complete

## Git Commit Guidelines

- Small, atomic commits
- Use conventional commits:
  - `feat:` - new feature
  - `fix:` - bug fix
  - `test:` - test updates
  - `docs:` - documentation
  - `refactor:` - code refactoring

## Collaboration Protocol

1. Coder implements one step
2. Coder runs tests
3. Verifier reviews code and tests
4. If OK → commit, move to next step
5. If NOT OK → Coder fixes, repeat
