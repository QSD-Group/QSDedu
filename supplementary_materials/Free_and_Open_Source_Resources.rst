==============================
Free and Open-Source Resources
==============================

A laundry list of other QSD-relevant Free and Open-Source (FOS) resources.


Process Modeling
----------------
Water/wastewater/sanitation
^^^^^^^^^^^^^^^^^^^^^^^^^^^
#. Water/wastewater treatment models (requires MATLAB) established/compiled by the `PROSYS center at the Technical University of Denmark <https://www.kt.dtu.dk/english/research/prosys>`_ and the `IEA division at the Lund University <https://iea.lth.se/>`_, you can find technical reports and more information about these models `here <https://wwtmodels.pubpub.org>`_, published models include (check the the website for the most-up-to-date list of models):

	#. `Activated sludge models <https://github.com/wwtmodels/Activated-Sludge-Models>`_
	#. `Anaerobic digestion models <https://github.com/wwtmodels/Anaerobic-Digestion-Models>`_
	#. `Benchmark simulation models <https://github.com/wwtmodels/Benchmark-Simulation-Models>`_
	#. `Biofilm models <https://github.com/wwtmodels/Biofilm-Models>`_
	#. `WWTP influent generators <https://github.com/wwtmodels/Influent-Generator-Models>`_
	#. `Plant-wide models <https://github.com/wwtmodels/Plant-Wide-Models>`_
	#. `System-wide models <https://github.com/wwtmodels/System-Wide-Models>`_
	
#. `Eawag <https://www.eawag.ch/en/>`_ `wastewater models <https://opendata.eawag.ch/organization/wastewater>`_, such as `aerobic granular sludge model <https://www.eawag.ch/en/department/eng/projects/abwasser/ags-aerobic-granular-slugde-model/>`_ (requires SUMO, includes R scripts for analysis/plotting and Python script for repetitive SUMO runs).
#. `Eawag <https://www.eawag.ch/en/>`_ `technology appropriateness assessment (TechAppA) model <https://github.com/Eawag-SWW/TechAppA>`_ for the quantification of the appropriateness of technologies (techs) in a given context (case) considering uncertainties
#. `Eawag <https://www.eawag.ch/en/>`_ `SWMM-HEAT model <https://github.com/Eawag-SWW/EAWAG-SWMM-HEAT>`_ for simulation of heat- and hydraulic processes in sewers.
#. `BIOMATH at Ghent University <https://biomath.ugent.be>`_'s `GitHub <https://github.com/UGentBiomath>`_ page.
#. `Water Technoeconomic Assessment Pipe-Parity Platform (WaterTAP3) <https://github.com/watertap-org/watertap>`_ developed under the National Alliance for Water Innovation (NAWI), currently focused on desalination.


Energy/Chemical Engineering
^^^^^^^^^^^^^^^^^^^^^^^^^^^
#. `IDAES Process Systems Engineering Framework <https://github.com/IDAES/idaes-pse>`_
#. `Waste-to-Energy System Simulation (WESyS) Model <https://github.com/NREL/WESyS-Model>`_


Curated Lists
-------------
#. `Industrial Ecology Dashborad <https://github.com/IndEcol/Dashboard>`_, the Dashboard includes some life cycle assessment (LCA) repositories, but some more LCA-focused ones:

	#. `calculator <https://github.com/romainsacchi/carculator>`_ for prospective environmental and economic life cycle assessment of vehicles
	#. Notes for a `Brightway2 <https://github.com/brightway-lca/brightway2>`_ (framework for advanced life cycle assessment calculations) `seminar <https://github.com/PoutineAndRosti/Brightway-Seminar-2017>`_
	
#. `Open Sustainable Technology <https://github.com/protontypes/open-sustainable-technology>`_
#. `Awesome Python Data Science <https://github.com/thomasjpfan/awesome-python-data-science>`_


Others
------
#. `Statistical Climate Forecasting Integrated for Civil and Environmental Engineering <https://github.com/yuchuan-lai/scifi>`_ (in R)



Databases
---------
#. `Our World in Data <https://ourworldindata.org/>`_, data sources on many topics.
#. Climate data:

	#. `GCM <https://www.ipcc-data.org/guidelines/pages/gcm_guide.html>`_ (general circulation models) projection results:

		#. `KNMI Climate Explorer <https://climexp.knmi.nl/start.cgi>`_
		#. `Earth System Grid Federation data <https://esgf-node.llnl.gov/projects/esgf-llnl/>`_

	#. Statistically downscaled GCM projections:

		#. `Localized constructed analogs <http://loca.ucsd.edu/>`_ (LOCA), CMIP5 (coupled model intercomparison project)
		#. `Multivariate adaptive constructed analogs <http://www.climatologylab.org/maca.html>`_ (MACA)-downscaled
		#. `Downscaled CMIP3 and CMIP5 climate and hydrology projections <https://gdo-dcp.ucllnl.org/downscaled_cmip_projections/dcpInterface.html>`_, multiple methods

	#. Dynamically downscaled GCM projections (more tools can be found at the `climate data gateway at National Center for Atmospheric Research <https://earthsystemgrid.org/>`_, NCAR):

		#. `North America Coordinated Regional Downscaling Experiment <https://earthsystemgrid.org/search/cordexsearch.html>`_ (NA-CORDEX)
		#. `North American Regional Climate Change Assessment Program <http://www.narccap.ucar.edu/>`_ (NARCCAP)

	#. Historical observations:
		
		#. `Global historical climatological network <http://scacis.rcc-acis.org/>`_
		#. `PRISM climate group <https://prism.oregonstate.edu/>`_ (parameter elevation regression on independent slopes model)
		#. `Climate at a glance <https://www.ncdc.noaa.gov/cag/>`_
		#. `Historical temperature and precipitation data for the U.S. cities <https://github.com/yuchuan-lai/Historical-City-ClimData>`_

#. `Drinking water treatability database <https://tdb.epa.gov/tdb/home>`_ by U.S. EPA.
#. U.S. `Federal LCA Commons <https://www.lcacommons.gov>`_, a central point of access to a collection of data repositories for use in LCA.
#. `Data.gov <https://www.data.gov>`_: home of the U.S. Government’s open data.
#. `Science.gov <https://www.science.gov>`_, gateway to U.S. government science information.
#. `Catalyst Property Database <https://cpd.chemcatbio.org>`_, currently contains DFT-computed adsorption energies for surface species in catalytic reactions. It is designed to reduce the time required to perform literature searches on previously computed catalytic pathways and intermediates by providing data in a central, searchable location.