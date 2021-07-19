JULIA = julia
JULIA16 = julia1.6
JULIA_OPTS = --startup-file=no

.PHONY: instantiate resolve update _jlpkg*

MANIFESTS = $(shell git ls-files | grep Manifest | xargs dirname)
JLPKG_TARGETS = $(patsubst %, _jlpkg-%, $(MANIFESTS)) 
JLPKG_COMMAND = error

instantiate:
	$(MAKE) JLPKG_COMMAND=instantiate _jlpkg

resolve:
	$(MAKE) JLPKG_COMMAND=resolve _jlpkg

update:
	$(MAKE) JLPKG_COMMAND=update _jlpkg

_jlpkg: $(JLPKG_TARGETS)

_jlpkg-test/environments/jl16:
	$(JULIA16) $(JUSLIA_CMD) -e 'using Pkg; Pkg.$(JLPKG_COMMAND)()' --project=test/environments/jl16

_jlpkg-docs:
	$(JULIA) $(JUSLIA_CMD) -e 'using Pkg; Pkg.$(JLPKG_COMMAND)()' --project=docs
