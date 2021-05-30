[![DOI](https://zenodo.org/badge/371717912.svg)](https://zenodo.org/badge/latestdoi/371717912)
# twitter-nordic-cross-border-mobility
Processing and analysis steps used in MSc thesis "Understanding Functional Cross-border Regions from Twitter Data in the Nordics". 

## Usage

Data used in this thesis is not availble in this repo. The Twitter data are collected using the Twitter API, to collect this data, look at the tool *tweetsearcher* created by Tuomo Väisänen: [https://github.com/DigitalGeographyLab/tweetsearcher](https://github.com/DigitalGeographyLab/tweetsearcher).

Data pre-processing and cleaning is found in *nordic_line_creation.py*, *nordic_data_cleaning.py*, and regions are assigned in *nordic_assign_regions.py*.

The Jupyter Notebook *nordic_cross_border_mobility.ipynb* lists out the steps of data exploration, temporal analyis, correlation analysis, and connectedness explorations.

General maps are created with *all_points_maps.py* and *all_lines_maps.py*. The calculations of functional cross-border regions can be found in *functional_area_maps.py*.

Images from the thesis is found in the *imgs* folder.

## Acknowledgements

The thesis and tools are developed in cooperation with the [Digital Geography Lab](https://www2.helsinki.fi/en/researchgroups/digital-geography-lab) at the University of Helsinki and as a part of the *BORDERSPACE -- Tracing Interactions and Mobilities Beyond State Borders: Towards New Transnational Spaces* [project](https://www2.helsinki.fi/en/researchgroups/digital-geography-lab/mobilities-and-interactions-of-people-crossing-state-borders-big-data-to-reveal-transnational-people-and-spaces).

Collection of data is based upon the tool *tweetsearcher* created by Tuomo Väisänen: [https://github.com/DigitalGeographyLab/tweetsearcher](https://github.com/DigitalGeographyLab/tweetsearcher)

Methodology builds upon the work of Samuli Massinen and his repo: [https://github.com/DigitalGeographyLab/cross-border-mobility-twitter](https://github.com/DigitalGeographyLab/cross-border-mobility-twitter)
