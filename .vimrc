set cin                    " C indent
set expandtab              " use space instead of tab
set softtabstop=2          " insert 3 spaces when pressing Tab
set shiftwidth=2           " insert 3 spaces for indent
set backspace=2            " powerful backspace
set mouse=a                " can use mouse
set showcmd                " show command
set ai                     " auto indent
set nu                     " show line number
set cursorline             " show cursorline
set tags=./tags,../tags    " ctags directory
syn on
filetype indent on

set background=dark

hi Normal ctermbg=Black
hi Visual ctermbg=Blue
hi Constant ctermfg=Cyan
hi Special ctermfg=Red
hi Comment ctermfg=Darkblue
