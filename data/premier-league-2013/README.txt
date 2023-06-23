Premier League (2013/2014) network, part of the Koblenz Network Collection
===========================================================================

This directory contains the TSV and related files of the league-uk1-2013 network: These are results of football games in England and Wales from the Premier League in the season 2013/2014, in form of a directed, signed graph.  Nodes are teams, and each directed edge from A to B denotes that team A played at home against team B.  The edge weights are the goal difference, and thus positive if the home team wins, negative when the away team wins, and zero for a draw.  The exact game results are not represented; only the goal differences are.  The data was copied by hand from Wikipedia. 


More information about the network is provided here: 
http://konect.cc/networks/league-uk1-2013

Files: 
    meta.league-uk1-2013 -- Metadata about the network 
    out.league-uk1-2013 -- The adjacency matrix of the network in whitespace-separated values format, with one edge per line
      The meaning of the columns in out.league-uk1-2013 are: 
        First column: ID of from node 
        Second column: ID of to node
        Third column (if present): weight or multiplicity of edge
        Fourth column (if present):  timestamp of edges Unix time
        Third column: edge weight


Use the following References for citation:

@MISC{konect:2018:league-uk1-2013,
    title = {Premier League (2013/2014) network dataset -- {KONECT}},
    month = mar,
    year = {2018},
    url = {http://konect.cc/networks/league-uk1-2013}
}

@article{konect:league,
	title = {Deformed {Laplacians} and Spectral Ranking in Directed Networks},
	author = {Michaël Fanuel and Johan A. K. Suykens},
	journal = {Applied and Computational Harmonic Analysis},
	year = {2017},
	note = {In press},
}


@inproceedings{konect,
	title = {{KONECT} -- {The} {Koblenz} {Network} {Collection}},
	author = {Jérôme Kunegis},
	year = {2013},
	booktitle = {Proc. Int. Conf. on World Wide Web Companion},
	pages = {1343--1350},
	url = {http://dl.acm.org/citation.cfm?id=2488173},
	url_presentation = {https://www.slideshare.net/kunegis/presentationwow},
	url_web = {http://konect.cc/},
	url_citations = {https://scholar.google.com/scholar?cites=7174338004474749050},
}

@inproceedings{konect,
	title = {{KONECT} -- {The} {Koblenz} {Network} {Collection}},
	author = {Jérôme Kunegis},
	year = {2013},
	booktitle = {Proc. Int. Conf. on World Wide Web Companion},
	pages = {1343--1350},
	url = {http://dl.acm.org/citation.cfm?id=2488173},
	url_presentation = {https://www.slideshare.net/kunegis/presentationwow},
	url_web = {http://konect.cc/},
	url_citations = {https://scholar.google.com/scholar?cites=7174338004474749050},
}


