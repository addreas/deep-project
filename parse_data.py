import pandas as pd 
import mwxml
import re

# REQUIREMENTS Pip install mwxml

articles_list = []

RE_P0 = re.compile(r'<!--.*?-->', re.DOTALL | re.UNICODE)
"""Comments."""
RE_P1 = re.compile(r'<ref([> ].*?)(</ref>|/>)', re.DOTALL | re.UNICODE)
"""Footnotes."""
RE_P2 = re.compile(r'(\n\[\[[a-z][a-z][\w-]*:[^:\]]+\]\])+$', re.UNICODE)
"""Links to languages."""
RE_P3 = re.compile(r'{{([^}{]*)}}', re.DOTALL | re.UNICODE)
"""Template."""
RE_P4 = re.compile(r'{{([^}]*)}}', re.DOTALL | re.UNICODE)
"""Template."""
RE_P5 = re.compile(r'\[(\w+):\/\/(.*?)(( (.*?))|())\]', re.UNICODE)
"""Remove URL, keep description."""
RE_P6 = re.compile(r'\[([^][]*)\|([^][]*)\]', re.DOTALL | re.UNICODE)
"""Simplify links, keep description."""
RE_P7 = re.compile(r'\n\[\[[iI]mage(.*?)(\|.*?)*\|(.*?)\]\]', re.UNICODE)
"""Keep description of images."""
RE_P8 = re.compile(r'\n\[\[[fF]ile(.*?)(\|.*?)*\|(.*?)\]\]', re.UNICODE)
"""Keep description of files."""
RE_P9 = re.compile(r'<nowiki([> ].*?)(</nowiki>|/>)', re.DOTALL | re.UNICODE)
"""External links."""
RE_P10 = re.compile(r'<math([> ].*?)(</math>|/>)', re.DOTALL | re.UNICODE)
"""Math content."""
RE_P11 = re.compile(r'<(.*?)>', re.DOTALL | re.UNICODE)
"""All other tags."""
RE_P12 = re.compile(r'(({\|)|(\|-(?!\d))|(\|}))(.*?)(?=\n)', re.UNICODE)
"""Table formatting."""
RE_P13 = re.compile(r'(?<=(\n[ ])|(\n\n)|([ ]{2})|(.\n)|(.\t))(\||\!)([^[\]\n]*?\|)*', re.UNICODE)
"""Table cell formatting."""
RE_P14 = re.compile(r'\[\[Category:[^][]*\]\]', re.UNICODE)
"""Categories."""
RE_P15 = re.compile(r'\[\[([fF]ile:|[iI]mage)[^]]*(\]\])', re.UNICODE)
"""Remove File and Image templates."""
RE_P16 = re.compile(r'\[{2}(.*?)\]{2}', re.UNICODE)
"""Capture interlinks text and article linked"""
RE_P17 = re.compile(
r'(\n.{0,4}((bgcolor)|(\d{0,1}[ ]?colspan)|(rowspan)|(style=)|(class=)|(align=)|(scope=))(.*))|'
r'(^.{0,2}((bgcolor)|(\d{0,1}[ ]?colspan)|(rowspan)|(style=)|(class=)|(align=))(.*))',
re.UNICODE)

dump = mwxml.Dump.from_file(open("Wikipedia-Dataset.xml"))
for page in dump:
	for revision in page:
		revision = revision.text.split('==References')[0]
		revision = revision.split('== References')[0]
		revision = revision.split('== Biobliography')[0]
		revision = revision.split('==Biobliography')[0]
		revision = revision.split('== Further reading')[0]
		revision = revision.split('==Further reading')[0]
		revision = revision.split('==See also')[0]
		revision = revision.split('== See also')[0]

		if '</div>' in revision:
			continue



		revision = re.sub(r"\[\[File.*\]\]\n", "", revision)  #Remove file links
		
		revision = re.sub(r"\[\[Image.*\]\]\n", "", revision)
		revision = re.sub(r"<gallery.*</gallery>","",revision,flags=re.DOTALL)

		
		revision = re.sub(RE_P1,"",revision) 	    		# Remove references

		revision = re.sub(r"\*.*?\n", "", revision)			# Remove lists
		revision = re.sub(RE_P0,"",revision) 				# Remove
		revision = re.sub(r"&[A-Za-z]+?;","", revision)     # Remove HTML
		

		revision = re.sub(r"\[\[.*?\|(.+?)\]\]",r"\1",revision)
		revision = re.sub(r"\{\{.*?\|(.+?)\|.*?\}\}",r"\1",revision)

		revision = re.sub(RE_P2,"",revision) 
		revision = re.sub(RE_P3,"",revision)
		revision = re.sub(RE_P4,"",revision) 
		revision = re.sub(RE_P5,"",revision)

		#revision = re.sub(RE_P7,"",revision)
		revision = re.sub(RE_P8,"",revision)


		revision = re.sub(RE_P9,"",revision) 
		revision = re.sub(RE_P10,"",revision) 

		revision = re.sub(RE_P11,"",revision)
		#revision = re.sub(RE_P12,"",revision)
		#revision = re.sub(RE_P13,"",revision) 

		revision = re.sub(RE_P14,"",revision)
		revision = re.sub(r'{\|.*?\|}',"",revision,flags=re.DOTALL)

		#revision = re.sub(RE_P15,"",revision)
		#revision = re.sub(RE_P16,"",revision) 
		#revision = re.sub(RE_P17,"",revision) 

		#revision = re.sub(r"\[\[([A-Za-z #\(\)]+\|)","",revision)
		#revision = re.sub(r"{{.*?}}", "", revision)
		revision = re.sub(r"[{}\[\]]*", "", revision) #Remove brackets

		revision = re.sub(r"\..*?\ ",". ",revision)
		revision = re.sub(r"filename=.*?\.[a-z0-9]+? ","",revision)


		revision = re.sub(r'(?m)^[#:;_].*?\n', '', revision) #Removes lines starting with #
		revision = re.sub(r'^\w\n',"", revision)

		revision = re.sub(r"[===|==][ ]?.+?[ ]?[===|==]\n+(=)",r"\1",revision)  # Remove unneccesary line

		revision = revision.strip()

		revision = re.sub(r'\n\n\n.*?\n\n\n',"",revision,flags=re.DOTALL)

		if len(revision) < 700:
			continue

		#print((revision))
		articles_list.append(revision)


		#print('--------------------------------------------------------------')