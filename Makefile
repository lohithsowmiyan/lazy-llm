#.   
#.   .-.
#.  (o o)     go ahead ...
#.  | O \     ... make my day
#.   \   \
#.    `~~~'

HUME       = <a href="http://github.com/timm/ezr">home</a>
CONTRIBUTE = <a href="https://github.com/timm/ezr/blob/main/CONRIBUTE.md">contribute</a>
LICENSE    = <a href="https://github.com/timm/ezr/blob/main/LICENSE.md">license</a>
ISSUES     = <a href="http://github.com/timm/ezr/issues">issues</a>

MENU       = $(HUME) | $(CONTRIBUTE) | $(ISSUES) | $(LICENSE)

IMAGE      = <img src="img/ezr.png" align=right width=150>
CSS        = p { text-align: right; } pre,code {font-size: x-small;}

#----------------------------------------------------------
SHELL     := bash 
MAKEFLAGS += --warn-undefined-variables
.SILENT:  

Root=$(shell git rev-parse --show-toplevel)
OUTPUT_FILE := var
GIT_MESSAGE := saving the output

help      :  ## show help
	awk 'BEGIN {FS = ":.*?## "; print "\nmake [WHAT]" } \
			/^[^[:space:]].*##/ {printf "   \033[36m%-18s\033[0m : %s\n", $$1, $$2} ' \
		$(MAKEFILE_LIST)
	awk 'sub(/#\. /,"") { printf "  \033[36m%-20s\033[0m \n", $$0}' Makefile
	
pull    : ## download
	git pull

push    : ## save
	git add $(OUTPUT_FILE)
	git commit -m "$(GIT_MESSAGE)"
	git push

name:
	read -p "word> " w; figlet -f mini -W $$w  | gawk '$$0 {print "#        "$$0}' |pbcopy

install   : ## install as  a local python package
	pip install -e  . --break-system-packages 

tests:
	-python3 -B ezr.py -R all; if [ $$? -eq 0 ];               \
	then printf "\n\033[1;32m==> PASSES\033[0m\n";              \
	     sed -i '' '1 s/failing-red/passing-green/' README.md;   \
	else printf "\n\033[1;31m==> FAILS\033[0m\n";                 \
			 sed -i '' '1 s/passing-green/failing-red/' README.md;     \
  fi

docs/%.html : %.py ## .py --> .html
	gawk -f etc/ab2ba.awk $< > docs/$<
	cd docs; pycco -d . $<; rm $<
	echo "$(CSS)" >> docs/pycco.css
	sed -i '' 's?<h1>?$(MENU)<hr>$(IMAGE)&?' $@
	@open $@

~/tmp/%.pdf: %.py  ## .py --> .pdf
	mkdir -p ~/tmp
	echo "pdf-ing $@ ... "
	a2ps                 \
		-Br                 \
		--chars-per-line 100  \
		--file-align=fill      \
		--line-numbers=1        \
		--borders=no             \
		--pro=color               \
		--left-title=""            \
		--columns  3                 \
		-M letter                     \
		--footer=""                    \
		--right-footer=""               \
	  -o	 $@.ps $<
	ps2pdf $@.ps $@; rm $@.ps    
	open $@

	

LOWS = data/config/SS-A.csv data/config/SS-B.csv data/config/SS-C.csv data/config/SS-E.csv data/config/SS-F.csv data/misc/auto93.csv \
	   data/misc/wc+wc-3d-c4-obj1.csv data/misc/wc+sol-3d-c4-obj1.csv lazy-llm/data/misc/wc+wc-3d-c4-obj1.csv data/config/SS-G.csv  \
	   data/config/SS-I.csv data/config/SS-J.csv
	   
MEDS = data/misc/rs-6d-c3_obj1.csv data/misc/rs-6d-c3_obj2.csv data/misc/sol-6d-c2-obj1.csv data/hpo/healthCloseIsses12mths0001-hard.csv \
       data/hpo/healthCloseIsses12mths0011-easy.csv data/config/SS-K.csv data/config/SS-L.csv data/process/pom3a.csv data/process/pom3b.csv data/process/pom3c.csv data/process/pom3d.csv \
	   data/config/SS-S.csv 

HIGHS = data/config/SS-M.csv data/config/SS-N.csv data/config/SS-O.csv data/config/SS-Q.csv data/config/SS-R.csv data/config/SS-U.csv \
        data/config/SS-X.csv data/config/SS-W.csv data/config/SS-T.csv data/config/X264_AllMeasurements.csv data/config/SQL_AllMeasurements.csv lazy-llm/data/config/Apache_AllMeasurements.csv \
		data/misc/HSMGP_num.csv data/process/coc1000.csv data/process/coc1000.csv data/process/xomo_flight.csv data/process/xomo_ground.csv lazy-llm/data/process/xomo_osp2.csv
		

LOWS_OUT  = $(patsubst data/%,var/out/lows/%,$(LOWS))
MEDS_OUT  = $(patsubst data/%,var/out/meds/%,$(MEDS))
HIGHS_OUT = $(patsubst data/%,var/out/highs/%,$(HIGHS))

# LOWS
var/out/lows/%.csv : data/%.csv
	@mkdir -p $(dir $@)
	echo $<
	python3 ./lazy.py --model lows --dataset $< | tee $@

# MEDS
var/out/meds/%.csv : data/%.csv
	@mkdir -p $(dir $@)
	echo $<
	python3 ./lazy.py --model meds --dataset $< | tee $@

# HIGHS
var/out/highs/%.csv : data/%.csv
	@mkdir -p $(dir $@)
	echo $<
	python3 ./lazy.py --model highs --dataset $< | tee $@

RQ123: 
	$(MAKE) -j $(LOWS_OUT)
	$(MAKE) -j $(MEDS_OUT)
	$(MAKE) -j $(HIGHS_OUT)


	




