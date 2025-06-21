# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile


# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)


# # Minimal makefile for Sphinx documentation

# # You can set these variables from the command line.
# SPHINXOPTS    ?=
# SPHINXBUILD   ?= sphinx-build
# SOURCEDIR     = .
# BUILDDIR      = _build
# APIDOC        ?= sphinx-apidoc
# MODULEPATH    = ../fuzzyops/

# # Put it first so that "make" without argument is like "make help".
# help:
# 	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# .PHONY: help Makefile

# # Generate API documentation
# apidoc:
# 	$(APIDOC) -o "$(SOURCEDIR)" "$(MODULEPATH)" "$(MODULEPATH)/tests" --force --separate --module-first

# # Build everything
# all: apidoc html

# # Build HTML after generating API docs
# html:
# 	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# # Clean build directory
# clean:
# 	rm -rf "$(BUILDDIR)"

# # Catch-all target
# %: Makefile
# 	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
