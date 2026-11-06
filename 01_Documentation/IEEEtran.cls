\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{IEEEtran}[2026/01/01 IEEE Conference Proceedings Format]
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{article}}
\ProcessOptions\relax
\LoadClass[10pt,twocolumn]{article}

\RequirePackage[a4paper,top=0.75in,bottom=0.8in,left=0.65in,right=0.65in,columnsep=0.22in]{geometry}
\RequirePackage{mathptmx}
\RequirePackage{helvet}
\RequirePackage{courier}

% Section numbering in Roman numerals like IEEEtran
\renewcommand{\thesection}{\Roman{section}}
\renewcommand{\thesubsection}{\Alph{subsection}}
\renewcommand{\thesubsubsection}{\arabic{subsubsection}}

\makeatletter
\renewcommand\section{\@startsection {section}{1}{\z@}%
                                   {-2.5ex \@plus -1ex \@minus -.2ex}%
                                   {1.2ex \@plus.2ex}%
                                   {\normalfont\normalsize\bfseries\scshape\centering}}
\renewcommand\subsection{\@startsection{subsection}{2}{\z@}%
                                     {-2.0ex\@plus -0.5ex \@minus -.2ex}%
                                     {0.8ex \@plus .2ex}%
                                     {\normalfont\normalsize\itshape}}
\renewcommand\subsubsection{\@startsection{subsubsection}{3}{\parindent}%
                                     {-1.5ex\@plus -0.5ex \@minus -.2ex}%
                                     {0.5ex \@plus .2ex}%
                                     {\normalfont\small\itshape}}
\makeatother

\providecommand{\IEEEkeywords}[1]{\vspace{0.5em}\noindent\textbf{\textit{Index Terms---}}#1\vspace{0.5em}}
\providecommand{\IEEEauthorblockN}[1]{\textbf{\large #1}\\}
\providecommand{\IEEEauthorblockA}[1]{\normalsize #1}
\providecommand{\appendices}{\appendix}

\renewenvironment{abstract}
 {\small\noindent\textbf{\textit{Abstract---}}\itshape}
 {\normalfont\vspace{0.5em}}

\setlength{\parindent}{1em}
\setlength{\parskip}{0pt}

\providecommand{\IEEEPARstart}[2]{\textbf{\Large #1}\textbf{#2}}
