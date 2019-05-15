import mwxml
import re
from string import printable as english

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

countUnder = 0
countOver = 0
file = mwxml.Dump.from_file(open("Wikipedia-Dataset.xml"))
for page in file:
	for revision in page:
		#print(revision.text)
		#continue

		revision = revision.text.split('==References')[0]
		revision = revision.split('== References')[0]
		revision = revision.split('== Biobliography')[0]
		revision = revision.split('==Biobliography')[0]
		revision = revision.split('== Further reading')[0]
		revision = revision.split('==Further reading')[0]
		revision = revision.split('==See also')[0]
		revision = revision.split('== See also')[0]
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
		revision = re.sub(r"&[A-Za-z]+?;","", revision)     # Remove HTML encoded words
		

		revision = re.sub(r"\[\[([^|]+?)\]\]",r'\1',revision)
		revision = re.sub(r"\[\[.*?\|(.+?)\]\]",r"\1",revision)
		#revision = re.sub(r"\{\{.*?\|(.+?)\|.*?\}\}",r"\1",revision)

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


		revision = re.sub(r"filename=.*?\.([a-z0-9]+?)?[ \n\r\t]","",revision)  #REmove filename


		revision = re.sub(r'(?m)^[#:;_| \t].*?\n', '', revision) #Removes lines starting with #,:,;,_,|

		revision = re.sub(r"[===|==][ ]?.+?[ ]?[===|==]\n+(=)",r"\1",revision)  # Remove unneccesary header line


		revision = re.sub(r"\..*?\ ",". ",revision)      #Remove characters after .
		revision = re.sub(r"\.\w+?\n",".\n",revision)

		revision = re.sub(r'\n\n\n.*?\n\n\n',"",revision,flags=re.DOTALL)

		revision = ''.join(c for c in revision if c in english)
		revision = revision.replace('|',' ')

		revision = re.sub(r"\w+?=\w+((=\w+)?)+","",revision) # Remove all words containing = except for titles

		revision = ''.join(c for c in revision.splitlines(1) if (len(c)>100 or c is '\n') or "==" in c)

		revision = revision.strip()


		if len(revision) < 700:
			countUnder = countUnder + 1
			continue

		countOver = countOver + 1
		#print((revision))
		#print('--------------------------------------------------------------')

		articles_list.append(revision)
#print(countUnder)
#print(countOver)

